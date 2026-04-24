#!/usr/bin/env python
"""
onepass_dense_mix_run.py  ·  stream → encode → FAISS → search

Variant of onepass_dense_run that generates code-mixed query embeddings
on the fly by interpolating two monolingual query vector sets.
"""

import argparse
import atexit
import gc
import logging
import os
import pathlib
import random
import re
import signal
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings
from transformers import BitsAndBytesConfig

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


E5_INSTRUCT_QUERY_PREFIX = "Instruct: Retrieve relevant passages.\nQuery: "


def detect_encoder_family(encoder: str) -> str:
    name = encoder.lower()
    if "jina-embeddings-v3" in name or "jina-embedding-v3" in name:
        return "jina-v3"
    if "qwen3-embedding" in name:
        return "qwen3"
    if "e5" in name and "instruct" in name:
        return "e5-instruct"
    return "default"


def query_encode_kwargs(encoder: str) -> Dict[str, str]:
    family = detect_encoder_family(encoder)
    if family == "qwen3":
        return {"prompt_name": "query"}
    if family == "jina-v3":
        return {"task": "retrieval.query"}
    return {}


def doc_encode_kwargs(encoder: str) -> Dict[str, str]:
    family = detect_encoder_family(encoder)
    if family == "jina-v3":
        return {"task": "retrieval.passage"}
    return {}


# ---------- helpers ----------
def batched(it: Iterable, n: int) -> Iterable[List]:
    buf: List = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def read_queries_tsv(
    path: pathlib.Path,
    qid_field: str,
    text_field: str,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if ln == 1 and len(parts) >= 2:
                head0, head1 = parts[0].lower(), parts[1].lower()
                if head0.startswith(qid_field.lower()) and head1.startswith(text_field.lower()):
                    continue
            if len(parts) < 2:
                raise SystemExit(f"[ERROR] Bad queries TSV line #{ln}: {line}")
            rows.append((parts[0], parts[1]))
    return rows


def encode_queries(
    model: SentenceTransformer,
    encoder: str,
    pool,
    qids: Sequence[str],
    texts: Sequence[str],
    batch_size: int,
    normalize: bool,
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}
    family = detect_encoder_family(encoder)
    encode_kwargs = query_encode_kwargs(encoder)
    for start in range(0, len(qids), batch_size):
        end = min(start + batch_size, len(qids))
        chunk_ids = qids[start:end]
        chunk_texts = texts[start:end]
        if family == "e5-instruct":
            chunk_texts = [f"{E5_INSTRUCT_QUERY_PREFIX}{t}" for t in chunk_texts]
        vecs = model.encode(
            chunk_texts,
            pool=pool,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            **encode_kwargs,
        )
        for qid, vec in zip(chunk_ids, vecs):
            outputs[qid] = vec.astype(np.float32, copy=False)
    return outputs


def encode_documents(
    model: SentenceTransformer,
    encoder: str,
    pool,
    texts: Sequence[str],
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    encode_kwargs = doc_encode_kwargs(encoder)
    return model.encode(
        list(texts),
        pool=pool,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        **encode_kwargs,
    )


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize_tag(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip("/"))
    return clean.strip("-") or "run"


DEFAULT_INDEX_ROOT = os.environ.get(
    "INDEX_ROOT",
    str(PROJECT_ROOT / "indexes" / "idx-mmarco-bge-m3"),
)


def default_query_cache_root(repo: str, encoder: str) -> pathlib.Path:
    dataset_tag = sanitize_tag(repo.split("/")[-1])
    encoder_tag = sanitize_tag(encoder.split("/")[-1])
    env_root = os.environ.get("QUERY_CACHE_ROOT")
    if env_root:
        return pathlib.Path(env_root)
    base = os.environ.get("QUERY_CACHE_ROOT_BASE", str(PROJECT_ROOT / "data"))
    return pathlib.Path(base) / f"enc-query-{dataset_tag}-{encoder_tag}"


def free_memory():
    """Best-effort release of large buffers / GPU cache."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def register_cleanup():
    atexit.register(free_memory)

    def _handler(signum, frame):
        free_memory()
        raise SystemExit(f"Received signal {signum}; shutting down.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def maybe_load_cached_queries(cache_dir: pathlib.Path, lang: str, qids: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
    cache_file = cache_dir / lang / "queries.npz"
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file)
        cached_qids = [str(x) for x in data["qids"].tolist()]
        if cached_qids != list(qids):
            logging.info(
                "Cached queries for %s at %s do not match requested qids; skipping cache.",
                lang,
                cache_file,
            )
            return None
        vecs = data["vecs"].astype(np.float32, copy=False)
        if vecs.shape[0] != len(qids):
            logging.info(
                "Cached queries for %s had mismatched shape (%d rows vs %d qids); skipping cache.",
                lang,
                vecs.shape[0],
                len(qids),
            )
            return None
        return {qid: vec for qid, vec in zip(qids, vecs)}
    except Exception as exc:  # pragma: no cover - best-effort cache
        logging.warning("Failed to load cached queries for %s: %s", lang, exc)
        return None


def save_query_cache(cache_dir: pathlib.Path, lang: str, qids: Sequence[str], vec_map: Dict[str, np.ndarray]):
    if not vec_map:
        return
    lang_dir = cache_dir / lang
    ensure_dir(lang_dir)
    ordered = [vec_map[qid] for qid in qids if qid in vec_map]
    np.savez_compressed(
        lang_dir / "queries.npz",
        qids=np.array(list(qids)),
        vecs=np.stack(ordered, axis=0),
    )


def maybe_load_cached_index(
    index_root: pathlib.Path, lang: str, expected_dim: Optional[int]
) -> Optional[Dict[str, object]]:
    lang_dir = index_root / lang
    index_path = lang_dir / "index.faiss"
    map_path = lang_dir / "docid_map.tsv"
    docids_path = lang_dir / "docids.txt"
    for p in (index_path, map_path, docids_path):
        if not p.exists():
            return None

    try:
        cached_index = faiss.read_index(str(index_path))
    except Exception as exc:  # pragma: no cover - runtime check
        logging.warning("Failed to read cached index for %s at %s: %s", lang, index_path, exc)
        return None

    if expected_dim and hasattr(cached_index, "d") and cached_index.d != expected_dim:
        logging.warning(
            "Cached index dim mismatch for %s: expected %s, found %s. Skipping cache.",
            lang,
            expected_dim,
            getattr(cached_index, "d", None),
        )
        return None

    base_index = cached_index.index if hasattr(cached_index, "index") else cached_index
    base_index = faiss.downcast_index(base_index)
    # Ensure reconstruct is supported before returning the cached index
    try:
        tmp = np.empty((base_index.d,), dtype=np.float32)
        base_index.reconstruct(0, tmp)
    except Exception as exc:
        logging.warning(
            "Cached index for %s base type %s does not support reconstruct; skipping cache. Error: %s",
            lang,
            type(base_index),
            exc,
        )
        return None

    return {
        "lang": lang,
        "index": cached_index,
        "base_index": base_index,
        "map_path": map_path,
    }


def parse_alpha_list(alpha_str: str) -> List[float]:
    if not alpha_str:
        raise SystemExit("--cm_alphas must contain at least one value.")
    alphas: List[float] = []
    for tok in alpha_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            alphas.append(float(tok))
        except ValueError as exc:
            raise SystemExit(f"[ERROR] Could not parse alpha '{tok}': {exc}") from exc
    if not alphas:
        raise SystemExit("No valid alpha values parsed from --cm_alphas.")
    return alphas


def format_alpha(alpha: float) -> str:
    if abs(alpha - round(alpha)) < 1e-8:
        return str(int(round(alpha)))
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_query_specs(
    query_tsv_args: Optional[Sequence[str]],
    q_primary: Optional[str],
    q_secondary: Optional[str],
) -> List[Tuple[str, pathlib.Path]]:
    specs: List[Tuple[str, pathlib.Path]] = []
    if query_tsv_args:
        for entry in query_tsv_args:
            if "=" not in entry:
                raise SystemExit(f"--query_tsv expects LANG=PATH, got '{entry}'.")
            lang, path = entry.split("=", 1)
            lang = lang.strip()
            path = path.strip()
            if not lang or not path:
                raise SystemExit(f"[ERROR] Bad --query_tsv entry '{entry}'.")
            specs.append((lang, pathlib.Path(path)))
    else:
        if not q_primary or not q_secondary:
            raise SystemExit("Provide either --query_tsv twice or both --q_en and --q_zh.")
        specs.append(("en", pathlib.Path(q_primary)))
        specs.append(("zh", pathlib.Path(q_secondary)))
    if len(specs) != 2:
        raise SystemExit(f"Exactly two query TSV specs required, got {len(specs)}.")
    seen = set()
    for lang, _ in specs:
        if lang in seen:
            raise SystemExit(f"Duplicate language '{lang}' in query specs.")
        seen.add(lang)
    return specs


def safe_mix(
    vec_primary: np.ndarray,
    vec_secondary: np.ndarray,
    alpha: float,
    qid: str,
    device: str,
    lang_pair: Optional[Tuple[str, str]] = None,
) -> np.ndarray:
    eps = 1e-8
    if abs(alpha) <= eps:
        return vec_primary
    if abs(alpha - 1.0) <= eps:
        return vec_secondary

    mixed = ((1.0 - alpha) * vec_primary + alpha * vec_secondary).astype(np.float32, copy=False)
    tensor = torch.from_numpy(mixed.reshape(1, -1)).to(dtype=torch.float32)
    try:
        target_device = torch.device(device)
    except Exception:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(target_device)
    normalized = normalize_embeddings(tensor).cpu().numpy().reshape(-1)
    if not np.all(np.isfinite(normalized)):
        fallback = vec_secondary if abs(alpha) > 0.5 else vec_primary
        if lang_pair:
            fallback_lang = lang_pair[1] if abs(alpha) > 0.5 else lang_pair[0]
        else:
            fallback_lang = "second" if abs(alpha) > 0.5 else "first"
        logging.warning(
            "Mixed embedding for qid=%s alpha=%.4f had non-finite values; using fallback vector (%s).",
            qid,
            alpha,
            fallback_lang,
        )
        return fallback
    return normalized.astype(np.float32, copy=False)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# ---------- main ----------
def main():
    register_cleanup()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # corpus & queries
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="collection")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--qrels_repo", default="BeIR/msmarco-qrels")
    ap.add_argument("--qrels_config", default="default")
    ap.add_argument("--qrels_split", default="test")
    ap.add_argument("--qrels_docid", default="corpus-id")
    ap.add_argument(
        "--query_tsv",
        action="append",
        metavar="LANG=PATH",
        help="Language-tagged query TSV (provide twice, e.g. 'id=/path/to/queries.id.tsv').",
    )
    ap.add_argument("--q_en", help="Legacy alias for --query_tsv en=PATH.")
    ap.add_argument("--q_zh", help="Legacy alias for --query_tsv zh=PATH.")
    ap.add_argument("--qid_field", default="id")
    ap.add_argument("--qtext_field", default="text")
    ap.add_argument(
        "--cm_alphas",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated alpha values (weight on the second language's vector).",
    )

    # runtime / size
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mp_devices")
    ap.add_argument("--gpu_faiss", action="store_true")
    ap.add_argument("--qblock", type=int, default=256, help="Query batch size for FAISS search")
    ap.add_argument(
        "--faiss_gpu_id",
        type=int,
        default=0,
        help="GPU id for FAISS when --gpu_faiss is set; uses visible index",
    )

    # model memory controls
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Preferred torch dtype for the encoder",
    )
    ap.add_argument("--load_in_8bit", action="store_true", help="Enable 8-bit quantization via bitsandbytes")
    ap.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit quantization via bitsandbytes (nf4)")
    ap.add_argument(
        "--attn_impl",
        default="auto",
        help="Attention implementation hint (e.g., flash_attention_2, sdpa, eager)",
    )
    ap.add_argument(
        "--max_memory_gib",
        type=float,
        help="Optional cap per visible GPU for HF device_map sharding, in GiB",
    )
    ap.add_argument("--neg_prob", type=float, default=0.02)
    ap.add_argument("--max_docs", type=int)
    ap.add_argument("--max_queries", type=int)

    # output
    ap.add_argument("--run_out", required=True, help="Directory to write TREC run files for each alpha.")
    ap.add_argument("--docids_out", required=True)
    ap.add_argument(
        "--index_root",
        default=DEFAULT_INDEX_ROOT,
        help="Parent directory containing per-language indexes. If present for the requested language, skip encoding.",
    )
    ap.add_argument(
        "--cache_queries",
        action="store_true",
        help="Cache normalized query encodings to disk for reuse.",
    )
    ap.add_argument(
        "--query_cache_dir",
        help="Optional override for query cache root; defaults to ./data/enc-query-<dataset>-<encoder> or $QUERY_CACHE_ROOT.",
    )

    # misc
    ap.add_argument("--trust_remote", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="global RNG seed for reproducibility")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.gpu_faiss and not hasattr(faiss, "StandardGpuResources"):
        logging.warning("FAISS GPU support not available; falling back to CPU index.")
        args.gpu_faiss = False
    doc_lang = args.config.replace("collection-", "")
    index_root = pathlib.Path(args.index_root) if args.index_root else None

    alphas = parse_alpha_list(args.cm_alphas)

    if getattr(args, "mp_devices", None):
        if getattr(args, "device", None):
            print(f"[onepass] --mp_devices provided; ignoring --device={args.device} to avoid double-loading.")
        visible = ",".join(d.split(":")[-1] for d in args.mp_devices.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible

        if args.dtype == "auto":
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        elif args.dtype == "bf16":
            dtype = torch.bfloat16
        elif args.dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(f"[onepass] dtype: {args.dtype} → {dtype}")

        qconf = None
        if args.load_in_4bit and args.load_in_8bit:
            raise ValueError("Use only one of --load_in_4bit or --load_in_8bit")
        if args.load_in_4bit:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            )
        elif args.load_in_8bit:
            qconf = BitsAndBytesConfig(load_in_8bit=True)

        max_mem = None
        if args.max_memory_gib:
            try:
                num_visible = len(visible.split(","))
                max_mem = {i: f"{args.max_memory_gib:.0f}GiB" for i in range(num_visible)}
            except Exception as exc:
                print(f"[onepass] Ignoring --max_memory_gib due to: {exc}")

        attn_impl = None if args.attn_impl == "auto" else args.attn_impl

        model = SentenceTransformer(
            args.encoder,
            device=None,
            trust_remote_code=getattr(args, "trust_remote", False),
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": dtype,
                **({"quantization_config": qconf} if qconf else {}),
                **({"max_memory": max_mem} if max_mem else {}),
                **({"attn_implementation": attn_impl} if attn_impl else {}),
            },
        )
        print(
            f"[onepass] Encoder={args.encoder} sharded across GPUs {args.mp_devices} with dtype={dtype}. "
            f"Quant: {('4-bit' if args.load_in_4bit else ('8-bit' if args.load_in_8bit else 'none'))}."
        )
    else:
        if args.dtype == "auto":
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        elif args.dtype == "bf16":
            dtype = torch.bfloat16
        elif args.dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(f"[onepass] dtype: {args.dtype} → {dtype}")
        qconf = None
        if args.load_in_4bit and args.load_in_8bit:
            raise ValueError("Use only one of --load_in_4bit or --load_in_8bit")
        if args.load_in_4bit:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            )
        elif args.load_in_8bit:
            qconf = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = None if args.attn_impl == "auto" else args.attn_impl

        model = SentenceTransformer(
            args.encoder,
            device=args.device,
            trust_remote_code=getattr(args, "trust_remote", False),
            model_kwargs={
                "torch_dtype": dtype,
                **({"quantization_config": qconf} if qconf else {}),
                **({"attn_implementation": attn_impl} if attn_impl else {}),
            },
        )
        print(f"[onepass] Encoder={args.encoder} loaded on device {args.device}.")

    pool = None
    dim = model.get_sentence_embedding_dimension()

    rel_ids = {
        str(r[args.qrels_docid])
        for r in load_dataset(
            args.qrels_repo,
            args.qrels_config,
            split=args.qrels_split,
            streaming=True,
            trust_remote_code=args.trust_remote,
        )
    }
    logging.info("Relevant ids harvested: %d", len(rel_ids))

    kept: List[str] = []
    rel_kept = 0
    neg_kept = 0

    cpu = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index = cpu
    id_lookup: Dict[int, str] = {}

    cached_index = None
    cache_hit = False
    RECON_BATCH = 100000
    LOG_INTERVAL = 100000
    if index_root and index_root.exists():
        logging.info("Index root exists at %s; trying cached index for %s", index_root, doc_lang)
        cached_index = maybe_load_cached_index(index_root, doc_lang, dim)
        if cached_index:
            cache_hit = True
            try:
                total_entries = max(
                    0, sum(1 for _ in cached_index["map_path"].open("r", encoding="utf-8")) - 1
                )
            except Exception:
                total_entries = 0
            logging.info(
                "Loaded cached index for %s from %s; skipping corpus encoding (est. %d vectors).",
                doc_lang,
                index_root / doc_lang,
                total_entries,
            )
            map_path = cached_index["map_path"]  # type: ignore[index]
            cached_faiss = cached_index["index"]  # type: ignore[index]
            logging.info("Loading docid map from %s", map_path)
            with map_path.open("r", encoding="utf-8") as fh:
                next(fh, None)  # header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        local_id = int(parts[0])
                    except ValueError:
                        continue
                    base_id = parts[2]
                    id_lookup[local_id] = str(base_id)
                    kept.append(str(base_id))
            rel_kept = sum(1 for b in kept if b in rel_ids)
            neg_kept = len(kept) - rel_kept
            logging.info(
                "Cached index ready for %s: %d vectors (relevant %d, non-relevant %d)",
                doc_lang,
                len(kept),
                rel_kept,
                neg_kept,
            )
            index = cached_faiss
        else:
            logging.info("No cached index found under %s; encoding corpus.", index_root / doc_lang)

    if args.gpu_faiss:
        res = faiss.StandardGpuResources()
        if cache_hit:
            index = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, index)
        else:
            index = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, cpu)
        print(f"[onepass] FAISS on GPU:{args.faiss_gpu_id}")

    corpus = None
    if not cache_hit:
        corpus = load_dataset(
            args.repo,
            args.config,
            split=args.split,
            streaming=True,
            trust_remote_code=args.trust_remote,
        )

    target_neg = args.max_docs or 0

    if not cache_hit:
        bar_total = len(rel_ids) + target_neg if target_neg else None
        bar = tqdm.tqdm(
            total=bar_total,
            unit="doc",
            desc="Encoding",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for batch in batched(corpus, args.batch):
            if target_neg and neg_kept >= target_neg and rel_kept >= len(rel_ids):
                break
            ids = [str(x[args.id_field]) for x in batch]
            texts = [x[args.text_field] for x in batch]

            keep_mask: List[bool] = []
            for doc_id in ids:
                if doc_id in rel_ids:
                    keep_mask.append(True)
                    rel_kept += 1
                elif neg_kept < target_neg and random.random() < args.neg_prob:
                    keep_mask.append(True)
                    neg_kept += 1
                else:
                    keep_mask.append(False)

            if not any(keep_mask):
                continue

            enc_ids = [doc_id for doc_id, keep in zip(ids, keep_mask) if keep]
            enc_texts = [text for text, keep in zip(texts, keep_mask) if keep]

            vecs = encode_documents(
                model,
                args.encoder,
                pool,
                enc_texts,
                args.batch,
                normalize=True,
            )
            add_ids = np.array([int(doc_id) for doc_id in enc_ids])
            index.add_with_ids(vecs, add_ids)
            for add_id, doc_id in zip(add_ids.tolist(), enc_ids):
                id_lookup[add_id] = str(doc_id)
            kept.extend(enc_ids)
            bar.update(len(enc_ids))
        bar.close()

        if args.gpu_faiss and cpu.ntotal > 0 and index is cpu:
            # ensure searches use GPU after building on CPU when cache not used
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, cpu)
            print(f"[onepass] FAISS on GPU:{args.faiss_gpu_id}")
    logging.info("Indexed %d rel + %d neg = %d total", rel_kept, neg_kept, len(kept))
    pathlib.Path(args.docids_out).write_text("\n".join(sorted(set(kept))))

    outdir = pathlib.Path(args.run_out)
    outdir.mkdir(parents=True, exist_ok=True)

    query_specs = parse_query_specs(args.query_tsv, args.q_en, args.q_zh)
    (primary_lang, primary_path), (secondary_lang, secondary_path) = query_specs

    primary_queries = read_queries_tsv(primary_path, args.qid_field, args.qtext_field)
    secondary_queries = read_queries_tsv(secondary_path, args.qid_field, args.qtext_field)

    primary_map = {qid: text for qid, text in primary_queries}
    secondary_map = {qid: text for qid, text in secondary_queries}

    common_qids = [qid for qid, _ in primary_queries if qid in secondary_map]
    missing_primary = sorted(set(secondary_map.keys()) - set(primary_map.keys()))
    missing_secondary = sorted(set(primary_map.keys()) - set(secondary_map.keys()))

    if missing_primary:
        logging.warning(
            "Skipping %d qids missing in %s file: first=%s",
            len(missing_primary),
            primary_lang,
            missing_primary[0],
        )
    if missing_secondary:
        logging.warning(
            "Skipping %d qids missing in %s file: first=%s",
            len(missing_secondary),
            secondary_lang,
            missing_secondary[0],
        )

    if not common_qids:
        raise SystemExit(
            f"No overlapping qids between query files for {primary_lang} and {secondary_lang}."
        )

    if args.max_queries:
        common_qids = common_qids[: args.max_queries]

    primary_texts = [primary_map[qid] for qid in common_qids]
    secondary_texts = [secondary_map[qid] for qid in common_qids]

    def normalize_map(vec_map: Dict[str, np.ndarray], device: str) -> None:
        if not vec_map:
            return
        keys = list(vec_map.keys())
        stacked = np.stack([vec_map[k] for k in keys], axis=0)
        tensor = torch.from_numpy(stacked).to(dtype=torch.float32)
        try:
            target_device = torch.device(device)
        except Exception:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(target_device)
        normalized = normalize_embeddings(tensor).cpu().numpy()
        for idx, key in enumerate(keys):
            vec_map[key] = normalized[idx]
    cache_root = (
        pathlib.Path(args.query_cache_dir)
        if args.query_cache_dir
        else default_query_cache_root(args.repo, args.encoder)
    )
    primary_raw_vecs: Optional[Dict[str, np.ndarray]] = None
    secondary_raw_vecs: Optional[Dict[str, np.ndarray]] = None

    if args.cache_queries:
        primary_raw_vecs = maybe_load_cached_queries(cache_root, primary_lang, common_qids)
        if primary_raw_vecs is not None:
            logging.info(
                "Loaded cached %s query encodings from %s",
                primary_lang,
                cache_root / primary_lang / "queries.npz",
            )
        secondary_raw_vecs = maybe_load_cached_queries(cache_root, secondary_lang, common_qids)
        if secondary_raw_vecs is not None:
            logging.info(
                "Loaded cached %s query encodings from %s",
                secondary_lang,
                cache_root / secondary_lang / "queries.npz",
            )

    if primary_raw_vecs is None:
        logging.info("Encoding %d %s queries (raw)…", len(common_qids), primary_lang)
        primary_raw_vecs = encode_queries(
            model,
            args.encoder,
            pool,
            common_qids,
            primary_texts,
            args.batch,
            normalize=False,
        )
        normalize_map(primary_raw_vecs, args.device or "cpu")
        if args.cache_queries:
            save_query_cache(cache_root, primary_lang, common_qids, primary_raw_vecs)

    if secondary_raw_vecs is None:
        logging.info("Encoding %d %s queries (raw)…", len(common_qids), secondary_lang)
        secondary_raw_vecs = encode_queries(
            model,
            args.encoder,
            pool,
            common_qids,
            secondary_texts,
            args.batch,
            normalize=False,
        )
        normalize_map(secondary_raw_vecs, args.device or "cpu")
        if args.cache_queries:
            save_query_cache(cache_root, secondary_lang, common_qids, secondary_raw_vecs)

    total_runs = 0
    eps = 1e-8
    for alpha in alphas:
        label = format_alpha(alpha)
        run_lines: List[str] = []
        q_cnt = 0

        # precompute query matrix for this alpha to enable batched FAISS search
        q_matrix = np.empty((len(common_qids), dim), dtype=np.float32)
        for idx, qid in enumerate(common_qids):
            if abs(alpha) <= eps:
                qvec = primary_raw_vecs[qid]
            elif abs(alpha - 1.0) <= eps:
                qvec = secondary_raw_vecs[qid]
            else:
                qvec = safe_mix(
                    primary_raw_vecs[qid],
                    secondary_raw_vecs[qid],
                    alpha,
                    qid,
                    args.device,
                    (primary_lang, secondary_lang),
                )
            q_matrix[idx] = qvec.astype(np.float32, copy=False)

        with tqdm.tqdm(
            total=len(common_qids),
            unit="q",
            desc=f"Searching (alpha={label})",
            leave=False,
        ) as search_bar:
            for start in range(0, len(common_qids), args.qblock):
                end = min(start + args.qblock, len(common_qids))
                q_chunk = q_matrix[start:end]
                D, I = index.search(q_chunk, 100)
                for row, qid in enumerate(common_qids[start:end]):
                    doc_ids = [id_lookup.get(int(doc), str(doc)) for doc in I[row]]
                    run_lines.extend(
                        f"{qid}\tQ0\t{doc}\t{rank}\t{score:.4f}\tonepass-cm"
                        for rank, (doc, score) in enumerate(zip(doc_ids, D[row]), 1)
                    )
                    q_cnt += 1
                search_bar.update(end - start)
        run_path = outdir / f"cm-alpha-{label}.trec"
        run_path.write_text("\n".join(run_lines))
        logging.info("Run saved: %s  (%d queries, alpha=%s)", run_path, q_cnt, label)
        total_runs += 1

    logging.info("Completed %d alpha settings.", total_runs)
    logging.info("Done ✓")


if __name__ == "__main__":
    main()

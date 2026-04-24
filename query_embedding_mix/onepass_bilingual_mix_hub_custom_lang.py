#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onepass_bilingual_mix_hub.py — bilingual one-pass retrieval with on-the-fly
query vector interpolation between two monolingual query sets.

This mirrors onepass_bilingual_hub.py for document handling, but replaces the
code-mixed query TSV inputs with dynamically generated embeddings obtained by
mixing the monolingual query vectors.
"""

import argparse
import atexit
import gc
import json
import logging
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------- logging ---------------------------

def setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# --------------------------- helpers ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def sanitize_tag(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip("/"))
    return clean.strip("-") or "run"


DEFAULT_INDEX_ROOT = os.environ.get(
    "INDEX_ROOT",
    str(PROJECT_ROOT / "indexes" / "idx-mmarco-bge-m3"),
)

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


def default_query_cache_root(repo: str, encoder: str) -> Path:
    dataset_tag = sanitize_tag(repo.split("/")[-1])
    encoder_tag = sanitize_tag(encoder.split("/")[-1])
    env_root = os.environ.get("QUERY_CACHE_ROOT")
    if env_root:
        return Path(env_root)
    base = os.environ.get("QUERY_CACHE_ROOT_BASE", str(PROJECT_ROOT / "data"))
    return Path(base) / f"enc-query-{dataset_tag}-{encoder_tag}"


def read_queries_tsv(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if (
                ln == 1
                and len(parts) >= 2
                and parts[0].lower().startswith("qid")
                and parts[1].lower().startswith("query")
            ):
                continue
            if len(parts) < 2:
                raise SystemExit(f"[ERROR] Bad queries TSV line #{ln}: {line}")
            rows.append((parts[0], parts[1]))
    return rows


def batched(iterable: Iterable, n: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def collapse_run_max(in_run: Path, out_run: Path):
    by_q = {}
    with open(in_run, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, did, _rk, sc, _tag = line.split()
            base = did.split("#", 1)[0]
            score = float(sc)
            by_q.setdefault(qid, {}).setdefault(base, []).append(score)
    with open(out_run, "w", encoding="utf-8") as out:
        for qid, groups in by_q.items():
            items = [(b, max(scores)) for b, scores in groups.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            for rank, (base, val) in enumerate(items, 1):
                out.write(f"{qid} Q0 {base} {rank} {val:.6f} bilingual-mix\n")


def encode_queries(
    model: SentenceTransformer,
    encoder: str,
    qids: Sequence[str],
    texts: Sequence[str],
    batch_size: int,
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
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            **encode_kwargs,
        ).astype(np.float32, copy=False)
        for qid, vec in zip(chunk_ids, vecs):
            outputs[qid] = vec
    return outputs


def torch_normalize_embeddings(vec_map: Dict[str, np.ndarray], device: str):
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


def encode_documents(
    model: SentenceTransformer,
    encoder: str,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    encode_kwargs = doc_encode_kwargs(encoder)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        **encode_kwargs,
    ).astype(np.float32, copy=False)


def maybe_load_cached_queries(cache_dir: Path, lang: str, qids: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
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


def save_query_cache(cache_dir: Path, lang: str, qids: Sequence[str], vec_map: Dict[str, np.ndarray]):
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


def maybe_load_cached_index(index_root: Path, lang: str, expected_dim: Optional[int]):
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
    # Ensure reconstruct is supported on the base index before returning
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
    values: List[float] = []
    for tok in alpha_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError as exc:
            raise SystemExit(f"[ERROR] Could not parse alpha '{tok}': {exc}") from exc
    if not values:
        raise SystemExit("No valid alpha values parsed from --cm_alphas.")
    return values


def format_alpha(alpha: float) -> str:
    if abs(alpha - round(alpha)) < 1e-8:
        return str(int(round(alpha)))
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_query_specs(
    query_tsv_args: Optional[Sequence[str]],
    q_primary: Optional[str],
    q_secondary: Optional[str],
) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    if query_tsv_args:
        for entry in query_tsv_args:
            if "=" not in entry:
                raise SystemExit(f"--query_tsv expects LANG=PATH, got '{entry}'.")
            lang, path = entry.split("=", 1)
            lang = lang.strip()
            path = path.strip()
            if not lang or not path:
                raise SystemExit(f"[ERROR] Bad --query_tsv entry '{entry}'.")
            specs.append((lang, Path(path)))
    else:
        if not q_primary or not q_secondary:
            raise SystemExit("Provide either --query_tsv twice or both --q_en and --q_zh.")
        specs.append(("en", Path(q_primary)))
        specs.append(("zh", Path(q_secondary)))
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


# --------------------------- main ------------------------------

def main():
    register_cleanup()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Corpus / qrels config (mirrors original script)
    ap.add_argument("--repo", required=True, help="HF dataset repo for corpus configs")
    ap.add_argument("--split", default="collection", help="Corpus split name")
    ap.add_argument("--qrels_repo", default="BeIR/msmarco-qrels", help="HF repo for qrels")
    ap.add_argument("--qrels_config", default="default", help="HF config/name for qrels")
    ap.add_argument("--qrels_split", default="validation")
    ap.add_argument("--qrels_docid", default="corpus-id", help="Column in qrels for document id")
    ap.add_argument("--trust_remote", action="store_true")

    # Encoder / runtime
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256, help="stream batch size when reading corpus")
    ap.add_argument("--enc_batch", type=int, default=16, help="encoder batch size for kept docs / queries")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max_docs", type=int, help="MAX NEGATIVES (not total); all relevant are included")
    ap.add_argument("--neg_prob", type=float, default=1.0, help="Probability to sample a negative (until max_docs)")

    ap.add_argument("--id_field", default="id")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # FAISS
    ap.add_argument("--gpu_faiss", action="store_true")
    ap.add_argument("--faiss_gpu_id", type=int, default=0)

    # Multi-language docs
    ap.add_argument("--langs", required=True, help="Comma-separated languages, e.g. 'en,zh'")

    # Query sources & mixing
    ap.add_argument(
        "--query_tsv",
        action="append",
        metavar="LANG=PATH",
        help="Language-tagged query TSV (provide twice, e.g. 'id=/path/to/queries.id.tsv')",
    )
    ap.add_argument("--q_en", help="Legacy alias for --query_tsv en=PATH")
    ap.add_argument("--q_zh", help="Legacy alias for --query_tsv zh=PATH")
    ap.add_argument(
        "--cm_alphas",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated alpha values (weight on the second language's vector)",
    )
    ap.add_argument("--max_queries", type=int, help="Optional limit on number of shared qids (after intersection)")

    # Outputs
    ap.add_argument("--outdir", required=True, help="Directory for run files / metadata")
    ap.add_argument("--docids_out", required=True)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--qblock", type=int, default=128)

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument(
        "--index_root",
        default=DEFAULT_INDEX_ROOT,
        help="Parent directory containing per-language indexes. If indexes for all requested languages exist, skip encoding and reuse them.",
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

    args = ap.parse_args()
    setup_logging(args.verbosity)
    random.seed(args.seed)
    if args.gpu_faiss and not hasattr(faiss, "StandardGpuResources"):
        logging.warning("FAISS GPU support not available; falling back to CPU index.")
        args.gpu_faiss = False

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    if not langs:
        raise SystemExit("No languages provided in --langs.")
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # 1) Harvest relevant base ids
    rel_ids: Set[str] = set()
    if args.qrels_repo and args.qrels_config and args.qrels_docid:
        logging.info(
            "Harvesting relevant ids from qrels: repo=%s config=%s split=%s",
            args.qrels_repo,
            args.qrels_config,
            args.qrels_split,
        )
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

    # 2) Load encoder
    logging.info("Loading encoder '%s' on %s", args.encoder, args.device)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    logging.info("Using dtype=%s", str(dtype))

    model = SentenceTransformer(
        args.encoder,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote", False),
        model_kwargs={"torch_dtype": dtype},
    )
    dim = model.get_sentence_embedding_dimension()
    logging.info("Encoder dimension: %d", dim)

    # 3) Build FAISS CPU index
    index_cpu = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    # bookkeeping
    next_id = 0
    id2doc: List[str] = []
    base_written: Set[str] = set()

    docids_out_path = Path(args.docids_out)
    ensure_dir(docids_out_path.parent)
    map_out_path = outdir / "docid_map.tsv"
    docids_out_tmp: List[str] = []
    index_root = Path(args.index_root) if args.index_root else None

    target_neg = args.max_docs or 0
    rel_missing: Set[str] = set(rel_ids)
    selected_bases: Set[str] = set()
    rel_kept_unique = 0
    neg_kept = 0
    kept_total = 0

    cached_indexes = []
    if index_root and index_root.exists():
        logging.info("Index root exists at %s; trying cached indexes for langs=%s", index_root, ",".join(langs))
        missing_langs = []
        for lang in langs:
            cached = maybe_load_cached_index(index_root, lang, dim)
            if cached is None:
                missing_langs.append(lang)
            else:
                cached_indexes.append(cached)
                logging.info("Cached index found for %s", lang)
        if missing_langs:
            logging.info(
                "Cached indexes missing for languages: %s. Falling back to on-the-fly encoding.",
                ", ".join(missing_langs),
            )
            cached_indexes = []
        elif cached_indexes:
            logging.info(
                "Loaded cached indexes for languages: %s. Skipping corpus encoding.",
                ", ".join([c["lang"] for c in cached_indexes]),
            )

    RECON_BATCH = 20000
    LOG_INTERVAL = 20000
    with open(map_out_path, "w", encoding="utf-8") as map_fh:
        print("derived_id\tbase_id\tlang", file=map_fh)

        if cached_indexes:
            for cached in cached_indexes:
                lang = cached["lang"]
                cached_map_path = cached["map_path"]
                cached_index = cached["index"]
                base_index = cached.get("base_index", cached_index)
                try:
                    total_entries = max(
                        0,
                        sum(1 for _ in cached_map_path.open("r", encoding="utf-8")) - 1,  # minus header
                    )
                except Exception:
                    total_entries = 0
                logging.info(
                    "Adding cached index for %s to combined index in batches of %d (est. %d vectors)",
                    lang,
                    RECON_BATCH,
                    total_entries,
                )
                logging.info("Opening map file %s", cached_map_path)

                batch_entries: List[Tuple[int, str, str]] = []
                total_added = 0
                with open(cached_map_path, "r", encoding="utf-8") as fh:
                    next(fh, None)  # header
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 3:
                            continue
                        try:
                            local_id = int(parts[0])
                        except ValueError:
                            continue
                        derived_id = parts[1]
                        base_id = parts[2]
                        batch_entries.append((local_id, derived_id, base_id))
                        if len(batch_entries) >= RECON_BATCH:
                            batch_entries.sort(key=lambda x: x[0])
                            vecs = np.vstack([base_index.reconstruct(e[0]) for e in batch_entries])
                            add_ids = np.arange(next_id, next_id + len(vecs), dtype=np.int64)
                            index_cpu.add_with_ids(vecs, add_ids)

                            for _, derived_id, base_id in batch_entries:
                                id2doc.append(derived_id)
                                print(f"{derived_id}\t{base_id}\t{lang}", file=map_fh)
                                if base_id not in base_written:
                                    docids_out_tmp.append(base_id)
                                    base_written.add(base_id)

                            kept_total += len(vecs)
                            rel_kept_unique += sum(1 for _, _, b in batch_entries if b in rel_ids)
                            next_id += len(vecs)
                            total_added += len(vecs)
                            if total_added // LOG_INTERVAL != (total_added - len(vecs)) // LOG_INTERVAL:
                                logging.info(
                                    "Cached %s progress: %d/%s vectors added",
                                    lang,
                                    total_added,
                                    total_entries if total_entries else "?",
                                )
                            batch_entries.clear()

                    if batch_entries:
                        batch_entries.sort(key=lambda x: x[0])
                        vecs = np.vstack([base_index.reconstruct(e[0]) for e in batch_entries])
                        add_ids = np.arange(next_id, next_id + len(vecs), dtype=np.int64)
                        index_cpu.add_with_ids(vecs, add_ids)

                        for _, derived_id, base_id in batch_entries:
                            id2doc.append(derived_id)
                            print(f"{derived_id}\t{base_id}\t{lang}", file=map_fh)
                            if base_id not in base_written:
                                docids_out_tmp.append(base_id)
                                base_written.add(base_id)

                        kept_total += len(vecs)
                        rel_kept_unique += sum(1 for _, _, b in batch_entries if b in rel_ids)
                        next_id += len(vecs)
                        total_added += len(vecs)

                neg_kept = kept_total - rel_kept_unique
                logging.info(
                    "Finished adding cached index for %s: %d vectors (relevant %d, non-relevant %d)",
                    lang,
                    total_added,
                    rel_kept_unique,
                    neg_kept,
                )
                # Drop references early so GC can reclaim memory once a language is done.
                cached_index = None
                base_index = None
                cached_map_path = None
                batch_entries = []
                free_memory()
            cached_indexes.clear()
            cached_indexes = []
            free_memory()
        else:
            for lang_idx, lang in enumerate(langs):
                cfg = f"collection-{lang}"
                logging.info("Streaming corpus: %s / %s (split=%s)", args.repo, cfg, args.split)
                stream = load_dataset(
                    args.repo,
                    cfg,
                    split=args.split,
                    streaming=True,
                    trust_remote_code=args.trust_remote,
                )

                remaining_for_lang: Optional[Set[str]] = None
                if lang_idx > 0:
                    remaining_for_lang = set(selected_bases)

                enc_total = None
                if lang_idx == 0:
                    t = len(rel_ids) + target_neg
                    enc_total = t if t > 0 else None
                else:
                    enc_total = len(selected_bases) if selected_bases else None
                lang_bar = tqdm(
                    total=enc_total,
                    unit="doc",
                    desc=f"Encode[{lang}]",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    leave=True,
                )

                for batch in batched(stream, args.batch):
                    if lang_idx == 0 and target_neg and neg_kept >= target_neg and not rel_missing:
                        break
                    if lang_idx > 0 and remaining_for_lang is not None and not remaining_for_lang:
                        break

                    ids: List[str] = []
                    texts: List[str] = []
                    for x in batch:
                        try:
                            base_id = str(x[args.id_field])
                            text = x.get(args.text_field, "")
                        except Exception:
                            continue
                        if not text:
                            continue
                        ids.append(base_id)
                        texts.append(text)

                    if not ids:
                        continue

                    keep_idx: List[int] = []
                    newly_selected_neg = 0
                    if lang_idx == 0:
                        remaining_neg_slots = max(0, target_neg - neg_kept)
                        for j, base_id in enumerate(ids):
                            if base_id in rel_ids:
                                keep_idx.append(j)
                                if base_id not in selected_bases:
                                    selected_bases.add(base_id)
                            elif target_neg and remaining_neg_slots > 0 and random.random() < args.neg_prob:
                                keep_idx.append(j)
                                if base_id not in selected_bases:
                                    selected_bases.add(base_id)
                                    newly_selected_neg += 1
                                    remaining_neg_slots -= 1
                    else:
                        for j, base_id in enumerate(ids):
                            if base_id in selected_bases:
                                keep_idx.append(j)

                    if not keep_idx:
                        continue

                    enc_ids = [ids[k] for k in keep_idx]
                    enc_texts = [texts[k] for k in keep_idx]

                    with torch.inference_mode():
                        vecs = encode_documents(
                            model,
                            args.encoder,
                            enc_texts,
                            args.enc_batch,
                        )

                    add_ids = np.arange(next_id, next_id + len(enc_ids), dtype=np.int64)
                    next_id += len(enc_ids)

                    for base_id in enc_ids:
                        derived = f"{base_id}#{lang}"
                        id2doc.append(derived)
                        print(f"{derived}\t{base_id}\t{lang}", file=map_fh)
                        if base_id not in base_written:
                            docids_out_tmp.append(base_id)
                            base_written.add(base_id)
                        if lang_idx == 0 and base_id in rel_missing:
                            rel_missing.remove(base_id)
                            rel_kept_unique += 1

                    index_cpu.add_with_ids(vecs, add_ids)
                    kept_total += len(enc_ids)
                    lang_bar.update(len(enc_ids))
                    if lang_idx == 0 and newly_selected_neg:
                        neg_kept += newly_selected_neg

                    if lang_idx > 0 and remaining_for_lang is not None:
                        for base_id in enc_ids:
                            remaining_for_lang.discard(base_id)

                    del vecs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                lang_bar.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

    logging.info(
        "Indexed rel_unique=%d, neg=%d, total_kept=%d", rel_kept_unique, neg_kept, kept_total
    )
    Path(args.docids_out).write_text("\n".join(sorted(set(docids_out_tmp))))
    logging.info("Docid map written: %s", str(map_out_path))
    logging.info("Docids (base) written: %s", str(Path(args.docids_out)))

    if index_cpu.ntotal == 0:
        raise SystemExit("No documents indexed. Check corpus fields and filters.")

    # 4) Load monolingual query sets
    query_specs = parse_query_specs(args.query_tsv, args.q_en, args.q_zh)
    (primary_lang, primary_path), (secondary_lang, secondary_path) = query_specs

    primary_rows = read_queries_tsv(primary_path)
    secondary_rows = read_queries_tsv(secondary_path)
    if not primary_rows or not secondary_rows:
        raise SystemExit("Both query files must be non-empty.")

    primary_map = {qid: text for qid, text in primary_rows}
    secondary_map = {qid: text for qid, text in secondary_rows}
    common_qids = [qid for qid, _ in primary_rows if qid in secondary_map]
    if not common_qids:
        raise SystemExit(
            f"No overlapping qids between query files for {primary_lang} and {secondary_lang}."
        )
    if args.max_queries:
        common_qids = common_qids[: args.max_queries]

    cache_root = (
        Path(args.query_cache_dir) if args.query_cache_dir else default_query_cache_root(args.repo, args.encoder)
    )
    primary_vecs: Optional[Dict[str, np.ndarray]] = None
    secondary_vecs: Optional[Dict[str, np.ndarray]] = None

    if args.cache_queries:
        primary_vecs = maybe_load_cached_queries(cache_root, primary_lang, common_qids)
        if primary_vecs is not None:
            logging.info(
                "Loaded cached %s query encodings from %s",
                primary_lang,
                cache_root / primary_lang / "queries.npz",
            )
        secondary_vecs = maybe_load_cached_queries(cache_root, secondary_lang, common_qids)
        if secondary_vecs is not None:
            logging.info(
                "Loaded cached %s query encodings from %s",
                secondary_lang,
                cache_root / secondary_lang / "queries.npz",
            )

    if primary_vecs is None:
        logging.info("Encoding %d %s queries…", len(common_qids), primary_lang)
        primary_vecs = encode_queries(
            model,
            args.encoder,
            common_qids,
            [primary_map[qid] for qid in common_qids],
            args.enc_batch,
        )
        torch_normalize_embeddings(primary_vecs, args.device)
        if args.cache_queries:
            save_query_cache(cache_root, primary_lang, common_qids, primary_vecs)

    if secondary_vecs is None:
        logging.info("Encoding %d %s queries…", len(common_qids), secondary_lang)
        secondary_vecs = encode_queries(
            model,
            args.encoder,
            common_qids,
            [secondary_map[qid] for qid in common_qids],
            args.enc_batch,
        )
        torch_normalize_embeddings(secondary_vecs, args.device)
        if args.cache_queries:
            save_query_cache(cache_root, secondary_lang, common_qids, secondary_vecs)

    alphas = parse_alpha_list(args.cm_alphas)
    encoded_sets: List[Tuple[str, List[str], np.ndarray]] = []
    for alpha in alphas:
        label = format_alpha(alpha)
        q_vectors: List[np.ndarray] = []
        for qid in common_qids:
            q_vectors.append(
                safe_mix(
                    primary_vecs[qid],
                    secondary_vecs[qid],
                    alpha,
                    qid,
                    args.device,
                    (primary_lang, secondary_lang),
                )
            )
        q_matrix = np.stack(q_vectors, axis=0).astype(np.float32, copy=False)
        encoded_sets.append((label, common_qids, q_matrix))
        logging.info("Prepared query vectors for alpha=%s (%d queries)", label, len(common_qids))

    # free encoder if not needed further
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # 5) Move index to GPU if requested
    index_search = index_cpu
    if args.gpu_faiss:
        res = faiss.StandardGpuResources()
        index_search = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, index_cpu)
        logging.info("[onepass] FAISS on GPU:%d", args.faiss_gpu_id)

    # 6) Search per alpha
    lang_id_a = re.sub(r"\s+", "_", primary_lang.strip())
    lang_id_b = re.sub(r"\s+", "_", secondary_lang.strip())
    tag = f"bilingual-mix-{lang_id_a}-{lang_id_b}"
    for label, qids, qvecs in encoded_sets:
        set_name = f"cm-alpha-{label}"
        run_raw = outdir / f"{set_name}_raw.trec"
        with open(run_raw, "w", encoding="utf-8") as out, tqdm(
            total=len(qvecs), desc=f"Searching FAISS ({set_name})", unit="q", leave=False
        ) as pbar:
            for i in range(0, len(qvecs), args.qblock):
                q_chunk = qvecs[i : i + args.qblock]
                sims, idxs = index_search.search(q_chunk, args.topk)
                for r, qid in enumerate(qids[i : i + args.qblock]):
                    row_scores = sims[r]
                    row_ids = idxs[r]
                    for rank, (sc, ix) in enumerate(zip(row_scores.tolist(), row_ids.tolist()), 1):
                        if ix < 0 or ix >= len(id2doc):
                            continue
                        did = id2doc[ix]
                        out.write(f"{qid} Q0 {did} {rank} {sc:.6f} {tag}\n")
                pbar.update(q_chunk.shape[0])

        run_base = outdir / f"{set_name}.trec"
        collapse_run_max(run_raw, run_base)

        meta = {
            "started_at": now_str(),
            "encoder": args.encoder,
            "device": args.device,
            "normalize": bool(args.normalize),
            "repo": args.repo,
            "split": args.split,
            "langs": langs,
            "alpha": label,
            "cm_alphas": args.cm_alphas,
            "query_langs": {
                primary_lang: str(primary_path),
                secondary_lang: str(secondary_path),
            },
            "docids_out": str(Path(args.docids_out)),
            "docid_map": str(map_out_path),
            "runs": {"raw": str(run_raw), "base": str(run_base)},
            "index": {
                "type": "IndexIDMap(IndexFlatIP)",
                "gpu": bool(args.gpu_faiss),
                "gpu_id": args.faiss_gpu_id if args.gpu_faiss else None,
                "size": int(index_cpu.ntotal),
                "dim": int(qvecs.shape[1]),
            },
            "topk": int(args.topk),
            "qblock": int(args.qblock),
            "rel_unique": int(rel_kept_unique),
            "neg_kept": int(neg_kept),
            "kept_total": int(kept_total),
            "max_queries": args.max_queries,
        }
        with open(outdir / f"{set_name}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logging.info("Completed set '%s' → %s , %s", set_name, run_raw.name, run_base.name)

    logging.info("All alpha settings completed. Outputs in: %s", str(outdir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
encode_multilingual_corpus.py

Encode corpus documents for all available languages once and save reusable
embeddings to disk. Mirrors the subset-selection logic from the custom
onepass scripts: if a subset cap is provided, the first language selects
relevant docs plus sampled negatives (up to the cap) and later languages
mirror that selection; otherwise, the full corpus is encoded for every
language.
Then it saves the embeddings in a FAISS index along with a mapping file
from integer ids used in the index to the original document ids.
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------- helpers ---------------------------
def setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def batched(iterable: Iterable, n: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def sanitize_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip("/"))
    return clean.strip("-") or "run"


def discover_langs(repo: str, trust_remote: bool) -> List[str]:
    configs = get_dataset_config_names(repo, trust_remote_code=trust_remote)
    langs = [c.replace("collection-", "") for c in configs if c.startswith("collection-")]
    if not langs:
        raise SystemExit(f"Could not discover languages for repo '{repo}'. Provide --langs.")
    return sorted(set(langs))


def detect_encoder_family(encoder: str) -> str:
    name = encoder.lower()
    if "jina-embeddings-v3" in name or "jina-embedding-v3" in name:
        return "jina-v3"
    if "qwen3-embedding" in name:
        return "qwen3"
    if "e5" in name and "instruct" in name:
        return "e5-instruct"
    return "default"


def doc_encode_kwargs(encoder: str) -> Dict[str, str]:
    family = detect_encoder_family(encoder)
    if family == "jina-v3":
        return {"task": "retrieval.passage"}
    return {}


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


def derive_save_dir(
    save_root: Path,
    repo: str,
    split: str,
    encoder: str,
    run_name: Optional[str],
    subset_cap: Optional[int],
) -> Path:
    ensure_dir(save_root)
    if run_name:
        base = sanitize_name(run_name)
    else:
        tag_repo = sanitize_name(repo.split("/")[-1])
        tag_enc = sanitize_name(encoder.split("/")[-1])
        tag_subset = f"-sub{subset_cap}" if subset_cap else ""
        base = f"idx-{tag_repo}-{split}-{tag_enc}{tag_subset}"
    out = save_root / base
    ensure_dir(out)
    return out


def load_existing_state(outdir: Path, langs: Sequence[str], first_lang: Optional[str]):
    """Load prior progress if resuming; collects per-language base ids."""
    base_ids_global: Set[str] = set()
    existing_langs: Set[str] = set()
    first_lang_selected: Set[str] = set()

    for lang in langs:
        lang_dir = outdir / lang
        map_path = lang_dir / "docid_map.tsv"
        if not map_path.exists():
            continue
        existing_langs.add(lang)
        with map_path.open("r", encoding="utf-8") as fh:
            header = next(fh, None)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                base_id = parts[-2] if len(parts) == 4 else parts[1]
                base_ids_global.add(base_id)
                if first_lang and lang == first_lang:
                    first_lang_selected.add(base_id)

        docids_path = lang_dir / "docids.txt"
        if docids_path.exists():
            for line in docids_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    base_ids_global.add(line.strip())
                    if first_lang and lang == first_lang:
                        first_lang_selected.add(line.strip())

    return base_ids_global, existing_langs, first_lang_selected


def load_rel_ids(args) -> Set[str]:
    rel_ids: Set[str] = set()
    if not args.subset_neg_cap:
        return rel_ids
    if not (args.qrels_repo and args.qrels_config and args.qrels_docid):
        raise SystemExit("Subset mode requires qrels to identify relevant documents.")
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
    return rel_ids


def parse_lang_path_specs(specs: Optional[Sequence[str]], arg_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not specs:
        return out
    for entry in specs:
        if "=" not in entry:
            raise SystemExit(f"{arg_name} expects LANG=PATH, got '{entry}'.")
        lang, raw_path = entry.split("=", 1)
        lang = lang.strip()
        raw_path = raw_path.strip()
        if not lang or not raw_path:
            raise SystemExit(f"[ERROR] Bad {arg_name} entry '{entry}'.")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise SystemExit(f"[ERROR] {arg_name} path for {lang} does not exist: {path}")
        if lang in out:
            raise SystemExit(f"[ERROR] Duplicate {arg_name} entry for language '{lang}'.")
        out[lang] = path
    return out


def load_docid_set(path: Path) -> Set[str]:
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            docid = raw.strip()
            if docid:
                out.add(docid)
    return out


# --------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Corpus config
    ap.add_argument("--repo", required=True, help="HF dataset repo; expects collection-<lang> configs.")
    ap.add_argument("--split", default="collection", help="Corpus split name within each config.")
    ap.add_argument(
        "--config",
        help="Optional explicit dataset config name. If set, this config is used for all requested langs.",
    )
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--trust_remote", action="store_true")

    # Encoder / batching
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256, help="Stream batch size when reading corpus.")
    ap.add_argument("--enc_batch", type=int, default=16, help="Encoder batch size.")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Preferred torch dtype for the encoder.",
    )

    # FAISS
    ap.add_argument("--gpu_faiss", action="store_true", help="Build index on GPU for faster adds.")
    ap.add_argument("--faiss_gpu_id", type=int, default=0, help="GPU id for FAISS when --gpu_faiss is set.")

    # Languages
    ap.add_argument(
        "--langs",
        help="Comma-separated languages. If omitted, discover configs named collection-<lang>.",
    )
    ap.add_argument(
        "--exclude_docids",
        action="append",
        metavar="LANG=PATH",
        help="Optional language-tagged base docid exclusion list. Matching ids are skipped before encoding.",
    )
    ap.add_argument(
        "--include_docids",
        action="append",
        metavar="LANG=PATH",
        help="Optional language-tagged base docid inclusion list. If set, only matching ids are encoded.",
    )

    # Subset control (mirrors onepass selection)
    ap.add_argument(
        "--subset_neg_cap",
        type=int,
        help="If set, sample up to this many negatives in the first language (relevants always kept); later languages mirror the selected base ids. If omitted, encode the full corpus for every language.",
    )
    ap.add_argument(
        "--neg_prob",
        type=float,
        default=1.0,
        help="Probability to keep a negative when sampling with --subset_neg_cap.",
    )
    ap.add_argument("--qrels_repo", default="BeIR/msmarco-qrels")
    ap.add_argument("--qrels_config", default="default")
    ap.add_argument("--qrels_split", default="validation")
    ap.add_argument("--qrels_docid", default="corpus-id")

    # Output
    default_save_root = os.environ.get(
        "INDEX_ROOT_BASE",
        str(PROJECT_ROOT / "indexes"),
    )
    ap.add_argument(
        "--save_root",
        default=default_save_root,
        help="Root directory for saved encodings (default: ./indexes or $INDEX_ROOT_BASE).",
    )
    ap.add_argument(
        "--run_name",
        help="Optional custom folder name under save_root. Defaults to a unique repo/split/encoder/timestamp tag.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbosity", type=int, default=1)

    args = ap.parse_args()
    setup_logging(args.verbosity)
    random.seed(args.seed)
    if args.gpu_faiss and not hasattr(faiss, "StandardGpuResources"):
        logging.warning("FAISS GPU support not available; falling back to CPU index.")
        args.gpu_faiss = False

    # Resolve languages
    if args.langs:
        langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    else:
        langs = discover_langs(args.repo, args.trust_remote)
    if not langs:
        raise SystemExit("No languages resolved; provide --langs or ensure repo has collection-<lang> configs.")
    logging.info("Languages: %s", ",".join(langs))

    subset_mode = bool(args.subset_neg_cap)
    rel_ids = load_rel_ids(args) if subset_mode else set()
    target_neg = args.subset_neg_cap or 0
    exclude_docid_specs = parse_lang_path_specs(args.exclude_docids, "--exclude_docids")
    include_docid_specs = parse_lang_path_specs(args.include_docids, "--include_docids")
    unknown_exclude_langs = sorted(set(exclude_docid_specs.keys()) - set(langs))
    if unknown_exclude_langs:
        raise SystemExit(
            f"[ERROR] --exclude_docids specified for languages not in --langs: {', '.join(unknown_exclude_langs)}"
        )
    unknown_include_langs = sorted(set(include_docid_specs.keys()) - set(langs))
    if unknown_include_langs:
        raise SystemExit(
            f"[ERROR] --include_docids specified for languages not in --langs: {', '.join(unknown_include_langs)}"
        )
    overlap_filter_langs = sorted(set(exclude_docid_specs.keys()) & set(include_docid_specs.keys()))
    if overlap_filter_langs:
        raise SystemExit(
            f"[ERROR] Do not specify both --exclude_docids and --include_docids for the same language(s): "
            f"{', '.join(overlap_filter_langs)}"
        )

    # Load encoder
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
        trust_remote_code=args.trust_remote,
        model_kwargs={"torch_dtype": dtype},
    )
    dim = model.get_sentence_embedding_dimension()

    save_root = Path(args.save_root).expanduser()
    outdir = derive_save_dir(
        save_root,
        args.repo,
        args.split,
        args.encoder,
        args.run_name,
        target_neg if subset_mode else None,
    )

    # Resume support: load prior work if it exists
    first_lang = langs[0]
    base_ids_global, existing_langs, first_lang_selected = load_existing_state(outdir, langs, first_lang)
    resuming = bool(existing_langs)
    logging.info(
        "%s index root: %s",
        "Resuming; existing languages=" + ",".join(sorted(existing_langs)) if resuming else "Saving",
        str(outdir),
    )

    # Subset bookkeeping; if resuming, carry over base ids selected in the first language
    selected_bases: Set[str] = set()
    if subset_mode and first_lang_selected:
        selected_bases = set(first_lang_selected)
    rel_missing: Set[str] = set(rel_ids) - selected_bases if subset_mode else set()
    neg_kept = 0

    per_lang_meta = []
    exclude_docid_counts: Dict[str, int] = {}
    include_docid_counts: Dict[str, int] = {}

    for lang_idx, lang in enumerate(langs):
        cfg = args.config if args.config else f"collection-{lang}"
        logging.info("Streaming corpus: %s / %s (split=%s)", args.repo, cfg, args.split)
        stream = load_dataset(
            args.repo,
            cfg,
            split=args.split,
            streaming=True,
            trust_remote_code=args.trust_remote,
        )

        lang_dir = outdir / lang
        ensure_dir(lang_dir)

        # Skip languages already completed (based on prior docid map / index)
        if (lang_dir / "index.faiss").exists():
            logging.info("Skipping language '%s' (already indexed).", lang)
            continue

        excluded_docids_path = exclude_docid_specs.get(lang)
        excluded_docids: Set[str] = set()
        if excluded_docids_path is not None:
            logging.info("Loading excluded docids for %s from %s", lang, excluded_docids_path)
            excluded_docids = load_docid_set(excluded_docids_path)
            logging.info("Excluded docids for %s: %d", lang, len(excluded_docids))
        exclude_docid_counts[lang] = len(excluded_docids)

        included_docids_path = include_docid_specs.get(lang)
        included_docids: Set[str] = set()
        if included_docids_path is not None:
            logging.info("Loading included docids for %s from %s", lang, included_docids_path)
            included_docids = load_docid_set(included_docids_path)
            logging.info("Included docids for %s: %d", lang, len(included_docids))
        include_docid_counts[lang] = len(included_docids)

        remaining_for_lang: Optional[Set[str]] = None
        if lang_idx > 0 and subset_mode:
            remaining_for_lang = set(selected_bases)

        enc_total = None
        if subset_mode:
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

        # Prepare per-language FAISS index
        index_cpu = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        index = index_cpu
        gpu_res = None
        if args.gpu_faiss:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, args.faiss_gpu_id, index_cpu)
            logging.info("FAISS moved to GPU:%d for indexing language '%s'.", args.faiss_gpu_id, lang)

        next_id = 0
        lang_rows: List[Tuple[int, str, str]] = []  # (int_id, derived_id, base_id)

        for batch in batched(stream, args.batch):
            if subset_mode and lang_idx == 0 and target_neg and neg_kept >= target_neg and not rel_missing:
                break
            if subset_mode and lang_idx > 0 and remaining_for_lang is not None and not remaining_for_lang:
                break

            ids: List[str] = []
            texts: List[str] = []
            for x in batch:
                try:
                    base_id = str(x[args.id_field])
                    text = x.get(args.text_field, "")
                except Exception:
                    continue
                if included_docids and base_id not in included_docids:
                    continue
                if base_id in excluded_docids:
                    continue
                if not text:
                    continue
                ids.append(base_id)
                texts.append(text)

            if not ids:
                continue

            keep_idx: List[int] = []
            newly_selected_neg = 0
            if not subset_mode:
                keep_idx = list(range(len(ids)))
            elif lang_idx == 0:
                remaining_neg_slots = max(0, target_neg - neg_kept)
                for j, base_id in enumerate(ids):
                    if base_id in rel_ids:
                        keep_idx.append(j)
                        if base_id not in selected_bases:
                            selected_bases.add(base_id)
                        continue
                    if target_neg and remaining_neg_slots > 0 and random.random() < args.neg_prob:
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

            index.add_with_ids(vecs, add_ids)

            for add_id, base_id in zip(add_ids.tolist(), enc_ids):
                derived = f"{base_id}#{lang}"
                lang_rows.append((add_id, derived, base_id))
                base_ids_global.add(base_id)
                if subset_mode and lang_idx == 0 and base_id in rel_missing:
                    rel_missing.remove(base_id)

            lang_bar.update(len(enc_ids))
            if subset_mode and lang_idx == 0 and newly_selected_neg:
                neg_kept += newly_selected_neg
            if subset_mode and lang_idx > 0 and remaining_for_lang is not None:
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

        # checkpoint index for this language (write CPU copy)
        if args.gpu_faiss:
            index_cpu = faiss.index_gpu_to_cpu(index)
        lang_index_path = lang_dir / "index.faiss"
        faiss.write_index(index_cpu, str(lang_index_path))

        # Write per-language mappings
        map_path = lang_dir / "docid_map.tsv"
        with map_path.open("w", encoding="utf-8") as fh:
            print("int_id\tderived_id\tbase_id\tlang", file=fh)
            for local_id, derived, base_id in sorted(lang_rows, key=lambda x: x[0]):
                print(f"{local_id}\t{derived}\t{base_id}\t{lang}", file=fh)

        docids_path = lang_dir / "docids.txt"
        docids_path.write_text(
            "\n".join(sorted({base for _, _, base in lang_rows})),
            encoding="utf-8",
        )

        (lang_dir / "meta.json").write_text(
            json.dumps(
                {
                    "lang": lang,
                    "count": int(next_id),
                    "dim": dim,
                    "subset_mode": subset_mode,
                    "index_path": str(lang_index_path),
                    "config": cfg,
                    "included_docids_path": str(included_docids_path) if included_docids_path else None,
                    "included_docids_count": include_docid_counts.get(lang, 0),
                    "excluded_docids_path": str(excluded_docids_path) if excluded_docids_path else None,
                    "excluded_docids_count": exclude_docid_counts.get(lang, 0),
                    "gpu_built": bool(args.gpu_faiss),
                    "faiss_gpu_id": args.faiss_gpu_id if args.gpu_faiss else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logging.info("Saved index and mappings for language '%s'.", lang)
        per_lang_meta.append(
            {
                "lang": lang,
                "count": int(next_id),
                "index_path": str(lang_index_path),
                "config": cfg,
                "included_docids_path": str(included_docids_path) if included_docids_path else None,
                "included_docids_count": include_docid_counts.get(lang, 0),
                "excluded_docids_path": str(excluded_docids_path) if excluded_docids_path else None,
                "excluded_docids_count": exclude_docid_counts.get(lang, 0),
            }
        )

    if not base_ids_global:
        raise SystemExit("No documents were indexed; check corpus and parameters.")

    Path(outdir / "docids.txt").write_text("\n".join(sorted(base_ids_global)), encoding="utf-8")

    meta = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "encoder": args.encoder,
        "device": args.device,
        "dtype": str(dtype),
        "normalize_embeddings": True,
        "repo": args.repo,
        "split": args.split,
        "config": args.config,
        "langs": langs,
        "subset_neg_cap": args.subset_neg_cap,
        "neg_prob": args.neg_prob,
        "rel_ids": len(rel_ids),
        "include_docids": {lang: str(path) for lang, path in include_docid_specs.items()},
        "include_docid_counts": include_docid_counts,
        "exclude_docids": {lang: str(path) for lang, path in exclude_docid_specs.items()},
        "exclude_docid_counts": exclude_docid_counts,
        "saved_dir": str(outdir),
        "dimension": dim,
        "indexes": per_lang_meta,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("All languages complete. Outputs: %s", str(outdir))


if __name__ == "__main__":
    main()

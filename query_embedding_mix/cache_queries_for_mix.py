#!/usr/bin/env python3
"""
cache_queries_for_mix.py

Pre-encode and cache query sets in the same layout the one-pass scripts use
when --cache_queries is supplied. If two query TSVs are provided, it intersects
their qids (ordered by the first TSV). If a single query TSV is provided, it
encodes that language's full qid list as-is. Encodings are normalized and saved
per language under:
    <query-cache-root>/<lang>/queries.npz

Use --cache_root to override the target directory.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------- model-specific query handling ----------------
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


def apply_query_format(encoder: str, texts: Sequence[str]) -> Tuple[Sequence[str], Dict[str, str]]:
    family = detect_encoder_family(encoder)
    kwargs: Dict[str, str] = {}
    if family == "e5-instruct":
        return [f"{E5_INSTRUCT_QUERY_PREFIX}{t}" for t in texts], kwargs
    if family == "qwen3":
        kwargs["prompt_name"] = "query"
        return texts, kwargs
    if family == "jina-v3":
        kwargs["task"] = "retrieval.query"
        return texts, kwargs
    return texts, kwargs


# ---------------- helpers ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize_tag(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip("/"))
    return clean.strip("-") or "run"


def default_query_cache_root(repo: str, encoder: str) -> Path:
    dataset_tag = sanitize_tag(repo.split("/")[-1])
    encoder_tag = sanitize_tag(encoder.split("/")[-1])
    env_root = os.environ.get("QUERY_CACHE_ROOT")
    if env_root:
        return Path(env_root)
    base = os.environ.get("QUERY_CACHE_ROOT_BASE", str(PROJECT_ROOT / "data"))
    return Path(base) / f"enc-query-{dataset_tag}-{encoder_tag}"


def read_queries_tsv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
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
    if len(specs) not in {1, 2}:
        raise SystemExit(f"Provide one or two query TSV specs, got {len(specs)}.")
    seen = set()
    for lang, _ in specs:
        if lang in seen:
            raise SystemExit(f"Duplicate language '{lang}' in query specs.")
        seen.add(lang)
    return specs


def encode_queries(
    model: SentenceTransformer,
    encoder: str,
    qids: Sequence[str],
    texts: Sequence[str],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}
    for start in range(0, len(qids), batch_size):
        end = min(start + batch_size, len(qids))
        chunk_ids = qids[start:end]
        chunk_texts = texts[start:end]
        chunk_texts, encode_kwargs = apply_query_format(encoder, chunk_texts)
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


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo name (used for cache path naming).")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--enc_batch", type=int, default=16, help="Encoder batch size for queries.")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Preferred torch dtype for the encoder.",
    )
    ap.add_argument("--trust_remote", action="store_true")
    ap.add_argument(
        "--query_tsv",
        action="append",
        metavar="LANG=PATH",
        help="Language-tagged query TSV (provide once or twice, e.g. 'id=/path/to/queries.id.tsv').",
    )
    ap.add_argument("--q_en", help="Legacy alias for --query_tsv en=PATH.")
    ap.add_argument("--q_zh", help="Legacy alias for --query_tsv zh=PATH.")
    ap.add_argument("--max_queries", type=int, help="Optional cap on number of shared qids (after intersection).")
    ap.add_argument(
        "--cache_root",
        help="Override cache directory; defaults to ./data/enc-query-<dataset>-<encoder> or $QUERY_CACHE_ROOT.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    # load query specs
    query_specs = parse_query_specs(args.query_tsv, args.q_en, args.q_zh)
    primary_lang, primary_path = query_specs[0]
    secondary_lang = None
    secondary_path = None
    if len(query_specs) == 2:
        secondary_lang, secondary_path = query_specs[1]

    primary_rows = read_queries_tsv(primary_path)
    if not primary_rows:
        raise SystemExit("Primary query file must be non-empty.")

    if secondary_path:
        secondary_rows = read_queries_tsv(secondary_path)
        if not secondary_rows:
            raise SystemExit("Secondary query file must be non-empty.")

        primary_map = {qid: text for qid, text in primary_rows}
        secondary_map = {qid: text for qid, text in secondary_rows}
        common_qids = [qid for qid, _ in primary_rows if qid in secondary_map]
        if not common_qids:
            raise SystemExit(
                f"No overlapping qids between query files for {primary_lang} and {secondary_lang}."
            )
        if args.max_queries:
            common_qids = common_qids[: args.max_queries]
        logging.info("Common qids: %d", len(common_qids))
    else:
        common_qids = [qid for qid, _ in primary_rows]
        if args.max_queries:
            common_qids = common_qids[: args.max_queries]
        logging.info("Using %d qids from %s", len(common_qids), primary_lang)

    # encoder setup
    if args.dtype == "auto":
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    logging.info("Loading encoder %s (dtype=%s) on %s", args.encoder, dtype, args.device)
    model = SentenceTransformer(
        args.encoder,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote", False),
        model_kwargs={"torch_dtype": dtype},
    )

    # encode + normalize
    primary_map = {qid: text for qid, text in primary_rows}
    logging.info("Encoding %d %s queries…", len(common_qids), primary_lang)
    primary_vecs = encode_queries(
        model,
        args.encoder,
        common_qids,
        [primary_map[qid] for qid in common_qids],
        args.enc_batch,
    )
    torch_normalize_embeddings(primary_vecs, args.device)

    secondary_vecs = None
    if secondary_lang and secondary_path:
        secondary_rows = read_queries_tsv(secondary_path)
        secondary_map = {qid: text for qid, text in secondary_rows}
        logging.info("Encoding %d %s queries…", len(common_qids), secondary_lang)
        secondary_vecs = encode_queries(
            model,
            args.encoder,
            common_qids,
            [secondary_map[qid] for qid in common_qids],
            args.enc_batch,
        )
        torch_normalize_embeddings(secondary_vecs, args.device)

    cache_root = Path(args.cache_root) if args.cache_root else default_query_cache_root(args.repo, args.encoder)
    logging.info("Saving caches to %s", cache_root)
    save_query_cache(cache_root, primary_lang, common_qids, primary_vecs)
    if secondary_lang and secondary_vecs is not None:
        save_query_cache(cache_root, secondary_lang, common_qids, secondary_vecs)
        logging.info("Done. Cached %d queries for %s and %s.", len(common_qids), primary_lang, secondary_lang)
    else:
        logging.info("Done. Cached %d queries for %s.", len(common_qids), primary_lang)


if __name__ == "__main__":
    main()

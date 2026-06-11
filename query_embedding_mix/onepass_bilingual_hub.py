#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onepass_bilingual_hub.py — bilingual one-pass retrieval (faithful to original onepass behavior)

Key points restored:
- Guarantee all qrels docids are encoded (at least one variant per base id)
- --max_docs means MAX NEGATIVES (not total cap), same as the older script
- Mask-first: only encode kept docs (relevant + sampled negatives) → small effective GPU batch
- Index: CPU IndexIDMap(IndexFlatIP) while encoding; clone to GPU after indexing if --gpu_faiss
- Streaming from HF exactly like before
- Derived ids: base#lang; collapse to base with fuse=max before eval
"""

import torch, os, re, sys, json, glob, gc, time, argparse, logging, random
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

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

def read_queries_tsv(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if ln == 1 and len(parts) >= 2 and parts[0].lower().startswith("qid") and parts[1].lower().startswith("query"):
                continue
            if len(parts) < 2:
                raise SystemExit(f"[ERROR] Bad queries TSV line #{ln}: {line}")
            rows.append((parts[0], parts[1]))
    return rows

def parse_set_name_from_file(qfile: Path) -> str:
    m = re.search(r"queries-(cm.+)$", qfile.stem)
    if m:
        return m.group(1)
    m = re.search(r"queries-(.+)$", qfile.stem)
    return m.group(1) if m else qfile.stem

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


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
    except Exception as exc:
        logging.warning("Failed to read cached index for %s at %s: %s", lang, index_path, exc)
        return None
    if expected_dim and getattr(cached_index, "d", None) != expected_dim:
        logging.warning(
            "Cached index dim mismatch for %s: expected %s, found %s. Skipping cache.",
            lang,
            expected_dim,
            getattr(cached_index, "d", None),
        )
        return None

    base_index = cached_index.index if hasattr(cached_index, "index") else cached_index
    base_index = faiss.downcast_index(base_index)
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

# --------------------------- collapse --------------------------

def collapse_run_max(in_run: Path, out_run: Path):
    by_q = {}
    with open(in_run, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            qid, _, did, _rk, sc, _tag = line.split()
            base = did.split("#", 1)[0]
            score = float(sc)
            by_q.setdefault(qid, {}).setdefault(base, []).append(score)
    with open(out_run, "w", encoding="utf-8") as out:
        for qid, groups in by_q.items():
            items = [(b, max(scores)) for b, scores in groups.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            for rank, (base, val) in enumerate(items, 1):
                out.write(f"{qid} Q0 {base} {rank} {val:.6f} bilingual-onepass\n")

# --------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Original-style args retained
    ap.add_argument("--repo", required=True, help="HF dataset repo for corpus & queries configs")
    ap.add_argument("--split", default="collection", help="Corpus split (same semantics as original)")
    ap.add_argument("--q_split", default="dev", help="Kept for compatibility; not used (queries are local TSVs)")
    ap.add_argument("--qrels_repo", default="BeIR/msmarco-qrels", help="HF dataset repo for qrels")
    ap.add_argument("--qrels_config", default="default", help="HF config/name for qrels")
    ap.add_argument("--qrels_split", default="validation")
    ap.add_argument("--qrels_docid", default="corpus-id", help="Column in qrels for document id")
    ap.add_argument("--trust_remote", action="store_true")

    ap.add_argument("--encoder", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256, help="stream-batch (HF read chunk)")
    ap.add_argument("--enc_batch", type=int, default=16, help="encoder batch size (kept subset)")
    ap.add_argument("--normalize", action="store_true")  # if you used this before, fine; we’ll also set normalize_embeddings=True
    ap.add_argument("--max_docs", type=int, help="MAX NEGATIVES (not total); all relevant are always included")
    ap.add_argument("--neg_prob", type=float, default=1.0, help="probability to keep a non-relevant doc as negative (until max_docs)")

    ap.add_argument("--id_field", default="id")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="Preferred torch dtype for the encoder")

    # FAISS
    ap.add_argument("--gpu_faiss", action="store_true")
    ap.add_argument("--faiss_gpu_id", type=int, default=0)

    # Multi-query
    ap.add_argument("--langs", required=True, help="Comma-separated, e.g. 'en,zh'")
    ap.add_argument("--q_directory", required=True)
    ap.add_argument("--q_glob", default="queries-cm*.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--docids_out", required=True)
    ap.add_argument("--index_root",
                    help="Optional path to cached FAISS index root (contains <lang>/index.faiss + docid_map.tsv).")
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--qblock", type=int, default=128)

    # Deprecated
    ap.add_argument("--run_out", help="DEPRECATED (ignored; use --outdir)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbosity", type=int, default=1)

    args = ap.parse_args()
    setup_logging(args.verbosity)
    random.seed(args.seed)
    if args.gpu_faiss and not hasattr(faiss, "StandardGpuResources"):
        logging.warning("FAISS GPU support not available; falling back to CPU index.")
        args.gpu_faiss = False

    if args.run_out:
        logging.warning("--run_out is deprecated and ignored. Use --outdir for multi-set outputs.")

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    if not langs:
        raise SystemExit("No languages provided in --langs.")
    outdir = Path(args.outdir); ensure_dir(outdir)

    # 1) Harvest relevant base ids (required for selection)
    rel_ids: Set[str] = set()
    if args.qrels_repo and args.qrels_config and args.qrels_docid:
        logging.info("Harvesting relevant ids from qrels: repo=%s config=%s split=%s",
                     args.qrels_repo, args.qrels_config, args.qrels_split)
        rel_ids = {
            str(r[args.qrels_docid]) for r in
            load_dataset(args.qrels_repo, args.qrels_config,
                         split=args.qrels_split, streaming=True,
                         trust_remote_code=args.trust_remote)
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
        device=args.device,  # avoid binding to a single device; enable sharding
        trust_remote_code=getattr(args, "trust_remote", False),
        model_kwargs={
            # "device_map": "auto",      # shard across all visible GPUs
            "torch_dtype": dtype,
            # **({"quantization_config": qconf} if qconf else {}),
            # **({"max_memory": max_mem} if max_mem else {}),
        },
    )
    dim = model.get_sentence_embedding_dimension()

    # 3) Build FAISS CPU index (IndexIDMap(IndexFlatIP))
    index_cpu = faiss.IndexIDMap(faiss.IndexFlatIP( # cosine if vectors normalized
        # dim will be inferred on first add_with_ids
        dim  # temporary placeholder; FAISS ignores for IDMap wrapper; real dim set on first add
    ))
    # NOTE: we’ll replace the inner index with correct dim on first add (see below)

    # bookkeeping
    next_id = 0
    id2doc: List[str] = []
    base_written: Set[str] = set()

    docids_out_path = Path(args.docids_out); ensure_dir(docids_out_path.parent)
    map_path = outdir / "docid_map.tsv"
    docids_out_tmp = []  # accumulate base ids (unique) then write once
    index_root = Path(args.index_root) if args.index_root else None

    # Selection counters (faithful to old semantics)
    target_neg = args.max_docs or 0      # MAX NEGATIVES
    rel_missing: Set[str] = set(rel_ids) # base ids we still need at least once
    selected_bases = set()        # all base ids selected (rel + neg)
    rel_kept_unique = 0
    neg_kept = 0
    kept_total = 0

    # Progress bars will be shown per-language to reflect all encodings

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

    with open(map_path, "w", encoding="utf-8") as map_fh:
        print("derived_id\tbase_id\tlang", file=map_fh)

        if cached_indexes:
            RECON_BATCH = 20000
            LOG_INTERVAL = 20000
            for cached in cached_indexes:
                lang = cached["lang"]
                cached_map_path = cached["map_path"]
                base_index = cached.get("base_index", cached["index"])
                try:
                    total_entries = max(
                        0,
                        sum(1 for _ in cached_map_path.open("r", encoding="utf-8")) - 1,
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
        else:
            for lang_idx, lang in enumerate(langs):
                cfg = f"collection-{lang}"
                logging.info("Streaming corpus: %s / %s (split=%s)", args.repo, cfg, args.split)
                stream = load_dataset(args.repo, cfg, split=args.split, streaming=True, trust_remote_code=args.trust_remote)
                # For subsequent languages, mirror exactly the base_ids selected by the first
                remaining_for_lang = None
                if lang_idx > 0:
                    remaining_for_lang = set(selected_bases)

                # Per-language encoding progress bar
                enc_total = None
                if lang_idx == 0:
                    t = len(rel_ids) + target_neg
                    enc_total = t if t > 0 else None
                else:
                    enc_total = len(selected_bases) if selected_bases else None
                lang_bar = tqdm(total=enc_total, unit="doc", desc=f"Encode[{lang}]",
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                                leave=True)

                # Stream in batches, apply keep-mask first (rel + sampled neg), then encode only kept
                for batch in batched(stream, args.batch):
                    # FIRST language: stop once quotas satisfied (all rel seen + neg quota)
                    if lang_idx == 0 and target_neg and neg_kept >= target_neg and not rel_missing:
                        break
                    # SUBSEQUENT languages: stop when we've mirrored all selected base ids
                    if lang_idx > 0 and remaining_for_lang is not None and not remaining_for_lang:
                        break

                    # Gather fields
                    ids   = []
                    texts = []
                    for x in batch:
                        try:
                            i = str(x[args.id_field])
                            t = x.get(args.text_field, "")
                        except Exception:
                            continue
                        if not t:
                            continue
                        ids.append(i)
                        texts.append(t)

                    if not ids:
                        continue

                    # Decide keep-mask like the old script
                    keep_idx = []
                    newly_selected_rel = 0
                    newly_selected_neg = 0
                    if lang_idx == 0:
                        # first language drives selection
                        remaining_neg_slots = max(0, target_neg - neg_kept)
                        for j, base_id in enumerate(ids):
                            if base_id in rel_ids:
                                keep_idx.append(j)  # always keep relevants
                                if base_id not in selected_bases:
                                    selected_bases.add(base_id)
                                    newly_selected_rel += 1
                                continue
                            # negatives only if requested and slots remain
                            if target_neg == 0 or remaining_neg_slots <= 0:
                                continue
                            if random.random() < args.neg_prob:
                                keep_idx.append(j)  # sampled negative
                                if base_id not in selected_bases:
                                    selected_bases.add(base_id)
                                    newly_selected_neg += 1
                                    remaining_neg_slots -= 1
                    else:
                        # SUBSEQUENT languages: include ONLY base_ids already selected in S
                        for j, base_id in enumerate(ids):
                            if base_id in selected_bases:
                                keep_idx.append(j)

                    if not keep_idx:
                        continue

                    enc_ids   = [ids[k]   for k in keep_idx]
                    enc_texts = [texts[k] for k in keep_idx]

                    # Encode only the KEPT subset (small, stable VRAM)
                    with torch.inference_mode():
                        vecs = model.encode(
                            enc_texts,
                            batch_size=args.enc_batch,         # critical: small encoder batch
                            convert_to_numpy=True,
                            normalize_embeddings=True,         # matches old code path
                            show_progress_bar=False
                        ).astype(np.float32, copy=False)

                    # Assign numeric ids and derived ids
                    add_ids = np.arange(next_id, next_id + len(enc_ids), dtype=np.int64)
                    next_id += len(enc_ids)

                    for base_id in enc_ids:
                        derived = f"{base_id}#{lang}"
                        id2doc.append(derived)
                        print(f"{derived}\t{base_id}\t{lang}", file=map_fh)

                        # write base id once for evaluate.py --filter_docids
                        if base_id not in base_written:
                            docids_out_tmp.append(base_id)
                            base_written.add(base_id)

                        # Update counters ONLY for the selection language
                        if lang_idx == 0:
                            # unique relevant base-ids
                            if base_id in rel_missing:
                                rel_missing.remove(base_id)
                                rel_kept_unique += 1

                    # Add to FAISS
                    index_cpu.add_with_ids(vecs, add_ids)

                    kept_total += len(enc_ids)
                    # Update per-language encoding bar with all encoded docs
                    lang_bar.update(len(enc_ids))
                    # Count negatives added this batch exactly once (selection language only)
                    if lang_idx == 0 and newly_selected_neg:
                        neg_kept += newly_selected_neg

                    # For subsequent languages, shrink the remaining set to mirror
                    if lang_idx > 0 and remaining_for_lang is not None:
                        for bid in enc_ids:
                            if bid in remaining_for_lang:
                                remaining_for_lang.remove(bid)

                    # Immediately drop transient allocations
                    del vecs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Do not break across languages: later languages must mirror the selection
                lang_bar.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

    # All language encoding bars are closed per-language
    logging.info("Indexed rel_unique=%d, neg=%d, total_kept=%d", rel_kept_unique, neg_kept, kept_total)

    # Write docids_out once (base ids, deduped)
    Path(args.docids_out).write_text("\n".join(docids_out_tmp))
    logging.info("Docid map written: %s", str(map_path))
    logging.info("Docids (base) written: %s", str(Path(args.docids_out)))

    if index_cpu.ntotal == 0:
        raise SystemExit("No documents indexed. Check corpus fields and filters.")

    # 4) Prepare query sets
    q_dir = Path(args.q_directory)
    if not q_dir.exists():
        raise SystemExit(f"Query directory not found: {q_dir}")
    q_files = sorted(Path(p) for p in glob.glob(str(q_dir / args.q_glob)))
    if not q_files:
        raise SystemExit(f"No query files matched pattern {args.q_glob} under {q_dir}")
    logging.info("Discovered %d query sets under %s", len(q_files), q_dir)

    # 5) Pre-encode all query sets BEFORE cloning index to GPU
    encoded_sets = []  # list of (set_name, qids, qvecs, set_dir)
    for qfile in q_files:
        set_name = parse_set_name_from_file(qfile)
        set_dir = Path(args.outdir) #/ set_name
        ensure_dir(set_dir)

        logging.info("Loading query set '%s' from %s", set_name, qfile.name)
        qrows = read_queries_tsv(qfile)
        if not qrows:
            logging.warning("Empty query file: %s (skipping)", qfile)
            continue
        qids = [qid for qid, _ in qrows]
        qtexts = [qt for _, qt in qrows]

        logging.info("Encoding %d queries for '%s'...", len(qids), set_name)
        with torch.inference_mode():
            qvecs = model.encode(
                qtexts,
                batch_size=args.enc_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        if qvecs.ndim == 1:
            qvecs = qvecs.reshape(1, -1)
        qvecs = qvecs.astype(np.float32, copy=False)
        encoded_sets.append((set_name, qids, qvecs, set_dir))

    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
        # (re-instantiating ST for queries keeps code symmetric but you can reuse `model` if you prefer)

    # 6) Move index to GPU (if requested) AFTER encoding
    index_search = index_cpu
    if args.gpu_faiss:
        res = faiss.StandardGpuResources()
        index_search = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, index_cpu)
        logging.info("[onepass] FAISS on GPU:%d", args.faiss_gpu_id)

    # 7) Search per set and write runs
    tag = "bilingual-onepass"
    for set_name, qids, qvecs, set_dir in encoded_sets:
        run_raw = set_dir / f"{set_name}_raw.trec"
        with open(run_raw, "w", encoding="utf-8") as out, tqdm(total=len(qvecs), desc=f"Searching FAISS ({set_name})", unit="q", leave=False) as pbar:
            for i in range(0, len(qvecs), args.qblock):
                q_chunk = qvecs[i:i+args.qblock]
                sims, idxs = index_search.search(q_chunk, args.topk)
                for r, qid in enumerate(qids[i:i+args.qblock]):
                    row_scores = sims[r]
                    row_ids = idxs[r]
                    for rank, (sc, ix) in enumerate(zip(row_scores.tolist(), row_ids.tolist()), 1):
                        if ix < 0 or ix >= len(id2doc):
                            continue
                        did = id2doc[ix]
                        out.write(f"{qid} Q0 {did} {rank} {sc:.6f} {tag}\n")
                pbar.update(q_chunk.shape[0])

        run_base = set_dir / f"{set_name}_base.trec"
        collapse_run_max(run_raw, run_base)

        meta = {
            "started_at": now_str(),
            "encoder": args.encoder,
            "device": args.device,
            "normalize": bool(args.normalize),
            "repo": args.repo,
            "split": args.split,
            "langs": langs,
            "qfile": str(qfile),
            "set_name": set_name,
            "docids_out": str(Path(args.docids_out)),
            "docid_map": str(map_path),
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
        }
        with open(set_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logging.info("Completed set '%s' → %s , %s", set_name, run_raw.name, run_base.name)

    logging.info("All query sets completed. Outputs in: %s", str(outdir))


if __name__ == "__main__":
    main()

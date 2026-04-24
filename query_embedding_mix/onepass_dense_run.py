#!/usr/bin/env python
"""
onepass_dense_run.py  ·  stream → encode → FAISS → search
✓ indexes all judged-relevant docs first
✓ stops reading once max_docs is reached
✓ ETA-aware progress bars
"""
import argparse, logging, pathlib, random, time, glob
from itertools import islice

import torch, os
import faiss, numpy as np, tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

# ---------- helpers ----------
def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf: yield buf


def resolve_doc_lang(config: str) -> str:
    if config.startswith("collection-"):
        return config.split("collection-", 1)[1]
    return config


def maybe_load_cached_index(index_root: pathlib.Path, lang: str, expected_dim: int):
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
    if getattr(cached_index, "d", None) != expected_dim:
        logging.warning(
            "Cached index dim mismatch for %s: expected %s, found %s. Skipping cache.",
            lang,
            expected_dim,
            getattr(cached_index, "d", None),
        )
        return None
    return {
        "index": cached_index,
        "map_path": map_path,
        "docids_path": docids_path,
    }

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # corpus & queries
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="collection")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--q_config", required=True)
    ap.add_argument("--q_split", default="dev")
    ap.add_argument("--qid_field", default="id")
    ap.add_argument("--qtext_field", default="text")
    ap.add_argument("--q_directory",
                    help="Directory containing multiple query files; when set, encodes corpus once and searches all files.")
    ap.add_argument("--q_glob", default="queries-cm*.tsv",
                    help="Glob pattern for query files under --q_directory (e.g., queries-cm*.tsv)")
    # ap.add_argument("--filter_qids",
    #             help="Path to TSV/CSV/TXT containing the common query ids. "
    #                  "Only these qids will be searched. Uses the first column per line.")
    # qrels
    ap.add_argument("--qrels_repo", default="BeIR/msmarco-qrels")
    ap.add_argument("--qrels_config", default="default")
    ap.add_argument("--q_file",  help="local TSV or JSONL id<tab>text")
    ap.add_argument("--qrels_split",  default="test")
    ap.add_argument("--qrels_docid",  default="corpus-id")
    # runtime / size
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mp_devices")
    ap.add_argument("--gpu_faiss", action="store_true")
    ap.add_argument("--faiss_gpu_id", type=int, default=0,
                    help="GPU id for FAISS when --gpu_faiss is set; uses visible index")
    # model memory controls
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="Preferred torch dtype for the encoder")
    ap.add_argument("--load_in_8bit", action="store_true",
                    help="Enable 8-bit quantization via bitsandbytes")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Enable 4-bit quantization via bitsandbytes (nf4)")
    ap.add_argument("--attn_impl", default="auto",
                    help="Attention implementation hint (e.g., flash_attention_2, sdpa, eager)")
    ap.add_argument("--max_memory_gib", type=float,
                    help="Optional cap per visible GPU for HF device_map sharding, in GiB")
    ap.add_argument("--neg_prob", type=float, default=0.02)
    ap.add_argument("--max_docs", type=int)
    ap.add_argument("--max_queries", type=int)
    # output
    ap.add_argument("--run_out", required=True)
    ap.add_argument("--docids_out", required=True)
    ap.add_argument("--index_root",
                    help="Optional path to cached FAISS index root (contains <lang>/index.faiss + docid_map.tsv).")
    # misc
    ap.add_argument("--trust_remote", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="global RNG seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.gpu_faiss and not hasattr(faiss, "StandardGpuResources"):
        logging.warning("FAISS GPU support not available; falling back to CPU index.")
        args.gpu_faiss = False

##OLDER CODE to load model into a single GPU
    # 1) model & (optional) multi-process pool
    # model = SentenceTransformer(
    #     args.encoder, 
    #     device=args.device, 
    #     # model_kwargs={
    #     #     "device_map": "auto",      # shard across all visible GPUs
    #     #     # "attn_implementation": "flash_attention_2"
    #     #     # "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    #     # },
    #     tokenizer_kwargs={"padding_side": "left"},
    # )
#########

######## not needed anymore ########
    # try:
    #     first = model._first_module()  # ST Transformer module
    #     am = getattr(first, "auto_model", None) or getattr(first, "model", None)
    #     if am is not None:
    #         am.to(dtype=torch.float16)
    #         print("[onepass] Cast encoder to fp16 for lower VRAM.")
    # except Exception as e:
    #     print(f"[onepass] fp16 cast skipped: {e}")
# ###################################

    if getattr(args, "mp_devices", None):
        if getattr(args, "device", None):
            print(f"[onepass] --mp_devices provided; ignoring --device={args.device} to avoid double-loading.")
        # e.g. args.mp_devices = 'cuda:0,cuda:1'
        visible = ",".join(d.split(":")[-1] for d in args.mp_devices.split(","))
        # Ensure only the requested GPUs are visible to this process
        os.environ["CUDA_VISIBLE_DEVICES"] = visible

        # Resolve dtype
        if args.dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif args.dtype == "bf16":
            dtype = torch.bfloat16
        elif args.dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(f"[onepass] dtype: {args.dtype} → {dtype}")

        # Optional quantization
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

        # Optional max_memory cap for each visible GPU (to leave headroom for FAISS)
        max_mem = None
        if args.max_memory_gib:
            try:
                num_visible = len(visible.split(","))
                max_mem = {i: f"{args.max_memory_gib:.0f}GiB" for i in range(num_visible)}
            except Exception as e:
                print(f"[onepass] Ignoring --max_memory_gib due to: {e}")

        # Attention implementation hint
        attn_impl = None if args.attn_impl == "auto" else args.attn_impl

        # IMPORTANT: when using device_map, set device=None and let HF sharding handle it
        model = SentenceTransformer(
            args.encoder,
            device=None,  # avoid binding to a single device; enable sharding
            trust_remote_code=getattr(args, "trust_remote", False),
            model_kwargs={
                "device_map": "auto",      # shard across all visible GPUs
                "torch_dtype": dtype,
                **({"quantization_config": qconf} if qconf else {}),
                **({"max_memory": max_mem} if max_mem else {}),
                **({"attn_implementation": attn_impl} if attn_impl else {}),
            },
        )
        print(f"[onepass] Encoder={args.encoder} sharded across GPUs {args.mp_devices} with dtype={dtype}. "
              f"Quant: {('4-bit' if args.load_in_4bit else ('8-bit' if args.load_in_8bit else 'none'))}.")
    else:
        # Single-device mode
        # Resolve dtype
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

    # IMPORTANT: Do NOT use SentenceTransformer's multi-process pool when sharding across GPUs.
    # It would spawn separate processes that each load the model, causing OOM and pickling errors
    # when accelerate inserts forward hooks. We therefore always keep pool=None.
    pool = None
    dim = model.get_sentence_embedding_dimension()

    doc_lang = resolve_doc_lang(args.config)
    index_root = pathlib.Path(args.index_root) if args.index_root else None
    id_lookup = {}
    kept = []
    cache_hit = False

    if index_root and index_root.exists():
        logging.info("Index root exists at %s; trying cached index for %s", index_root, doc_lang)
        cached = maybe_load_cached_index(index_root, doc_lang, dim)
        if cached:
            cache_hit = True
            index = cached["index"]
            logging.info(
                "Loaded cached index for %s from %s; skipping corpus encoding.",
                doc_lang,
                index_root / doc_lang,
            )
            with cached["map_path"].open("r", encoding="utf-8") as fh:
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
            kept = cached["docids_path"].read_text(encoding="utf-8").splitlines()
            pathlib.Path(args.docids_out).write_text("\n".join(kept))
        else:
            logging.info("No cached index found under %s; encoding corpus.", index_root / doc_lang)

    if not cache_hit:
        # 2) gather judged-relevant ids
        rel_ids = {str(r[args.qrels_docid]) for r in
                   load_dataset(args.qrels_repo, args.qrels_config,
                                split=args.qrels_split, streaming=True,
                                trust_remote_code=args.trust_remote)}
        logging.info("Relevant ids harvested: %d", len(rel_ids))

        # 3) FAISS
        cpu = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        if args.gpu_faiss:
            # Use the requested FAISS GPU id relative to CUDA_VISIBLE_DEVICES
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, cpu)
            print(f"[onepass] FAISS on GPU:{args.faiss_gpu_id}")
        else:
            index = cpu

        # 4) stream corpus
        # skip this part if already encoded and output is found at out_dir
        corpus = load_dataset(args.repo, args.config, split=args.split,
                              streaming=True, trust_remote_code=args.trust_remote)

        target_neg = args.max_docs or 0
        kept, rel_kept, neg_kept = [], 0, 0
        bar_total = len(rel_ids) + target_neg if target_neg else None
        bar = tqdm.tqdm(total=bar_total, unit="doc", desc="Encoding",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}]")

        for batch in batched(corpus, args.batch):
            # EARLY BREAK once quota satisfied ------------------------------
            if target_neg and neg_kept >= target_neg:
                # continue streaming *only* if we still miss some relevant docs
                if rel_kept >= len(rel_ids):
                    break
            # ----------------------------------------------------------------
            ids   = [str(x[args.id_field]) for x in batch]
            texts = [x[args.text_field]    for x in batch]

            keep_mask = []
            for i in ids:
                if i in rel_ids:
                    keep_mask.append(True);  rel_kept += 1
                elif neg_kept < target_neg and random.random() < args.neg_prob:
                    keep_mask.append(True);  neg_kept += 1
                else:
                    keep_mask.append(False)

            if not any(keep_mask):
                continue

            enc_ids   = [i for i, k in zip(ids, keep_mask)  if k]
            enc_texts = [t for t, k in zip(texts, keep_mask) if k]

            vecs = model.encode(enc_texts, pool=pool, batch_size=args.batch,
                                convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=False)
            add_ids = np.array([int(i) for i in enc_ids])
            index.add_with_ids(vecs, add_ids)
            for add_id, doc_id in zip(add_ids.tolist(), enc_ids):
                id_lookup[add_id] = str(doc_id)
            kept.extend(enc_ids)
            bar.update(len(enc_ids))
        bar.close()
        logging.info("Indexed %d rel + %d neg = %d total",
                     rel_kept, neg_kept, len(kept))
        pathlib.Path(args.docids_out).write_text("\n".join(kept))

    if cache_hit and args.gpu_faiss:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, args.faiss_gpu_id, index)
        print(f"[onepass] FAISS on GPU:{args.faiss_gpu_id}")

    # 5) search
    trec, q_cnt = [], 0

    # allowed_qids = None
    # if args.filter_qids:
    #     path = pathlib.Path(args.filter_qids)
    #     lines = path.read_text(encoding="utf-8").splitlines()
    #     # robust: grab first whitespace/TSV field as qid
    #     allowed_qids = {line.strip().split()[0] for line in lines if line.strip()}
    #     logging.info("Restricting search to %d common qids from %s",
    #              len(allowed_qids), path)

    # Support three modes:
    #  (A) --q_directory set: search over all matching files; write one run per file under --run_out (as a directory)
    #  (B) --q_file set     : search a single local file and write to --run_out (as a file)
    #  (C) else             : stream queries from HF dataset args.q_config and write to --run_out

    if args.q_directory and args.q_file:
        raise ValueError("Provide only one of --q_directory or --q_file, not both.")

    if args.q_directory:
        outdir = pathlib.Path(args.run_out)
        outdir.mkdir(parents=True, exist_ok=True)
        q_dir = pathlib.Path(args.q_directory)
        q_files = sorted(pathlib.Path(p) for p in glob.glob(str(q_dir / args.q_glob)))
        q_files = [qf for qf in q_files if "qids-common" not in qf.name]
        if not q_files:
            raise SystemExit(f"No query files matched pattern {args.q_glob} under {q_dir}")
        logging.info("Discovered %d query files under %s", len(q_files), q_dir)

        import re
        def parse_set_name_from_file(qfile: pathlib.Path) -> str:
            m = re.search(r"queries-(cm.+)$", qfile.stem)
            if m:
                return m.group(1)
            m = re.search(r"queries-(.+)$", qfile.stem)
            return m.group(1) if m else qfile.stem

        total_q = 0
        for qf in q_files:
            # load per-file queries (TSV or JSONL)
            if qf.suffix.lower() == ".tsv":
                import pandas as pd
                df = pd.read_csv(qf, sep="\t", names=[args.qid_field, args.qtext_field])
                queries = (dict(df.iloc[i]) for i in range(len(df)))
            else:  # assume JSONL
                queries = load_dataset('json', data_files=str(qf),
                                       split='train', streaming=True)

            set_name = parse_set_name_from_file(qf)
            trec = []
            q_cnt = 0
            for rec in tqdm.tqdm(queries, desc=f"Searching: {qf.name}", unit="qry"):
                if args.max_queries and q_cnt >= args.max_queries: break
                qid, txt = str(rec[args.qid_field]), rec[args.qtext_field]
                qvec = model.encode([txt], pool=pool, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
                D, I = index.search(qvec, 100)
                doc_ids = [id_lookup.get(int(doc), str(doc)) for doc in I[0]]
                trec.extend(f"{qid}\tQ0\t{doc}\t{rank}\t{score:.4f}\tonepass"
                            for rank, (doc, score) in enumerate(zip(doc_ids, D[0]), 1))
                q_cnt += 1

            run_path = outdir / f"{set_name}.trec"
            run_path.write_text("\n".join(trec))
            total_q += q_cnt
            logging.info("Run saved: %s  (%d queries)", str(run_path), q_cnt)

        logging.info("Completed %d query files. Total queries: %d", len(q_files), total_q)

    else:
        if args.q_file:
            if args.q_file.endswith(".tsv"):
                import pandas as pd
                df = pd.read_csv(args.q_file, sep="\t", names=[args.qid_field,
                                                               args.qtext_field])
                queries = (dict(df.iloc[i]) for i in range(len(df)))
            else:   # assume JSONL
                queries = load_dataset('json', data_files=args.q_file,
                                       split='train', streaming=True)
        else:
            queries = load_dataset(args.repo, args.q_config,
                                   split=args.q_split, streaming=True,
                                   trust_remote_code=args.trust_remote)

        for rec in tqdm.tqdm(queries, desc="Searching", unit="qry"):
            if args.max_queries and q_cnt >= args.max_queries: break
            qid, txt = str(rec[args.qid_field]), rec[args.qtext_field]
            qvec = model.encode([txt], pool=pool, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
            D, I = index.search(qvec, 100)
            doc_ids = [id_lookup.get(int(doc), str(doc)) for doc in I[0]]
            trec.extend(f"{qid}\tQ0\t{doc}\t{rank}\t{score:.4f}\tonepass"
                        for rank, (doc, score) in enumerate(zip(doc_ids, D[0]), 1))
            q_cnt += 1
        pathlib.Path(args.run_out).write_text("\n".join(trec))
        logging.info("Run saved: %s  (%d queries)", args.run_out, q_cnt)

    # No pool used; nothing to stop
    logging.info("Done ✓")

if __name__ == "__main__":
    main()

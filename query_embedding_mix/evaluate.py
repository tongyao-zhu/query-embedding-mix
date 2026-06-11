#!/usr/bin/env python
"""
evaluate.py
CLI examples
------------
Local qrels    : python evaluate.py ---dataset <name> --run <path/to/run.trec>
Hub-stream qrels: python evaluate.py --run my.trec \
                    --qrels_repo BeIR/msmarco-qrels --qrels_split test
"""
import argparse, datetime, json, os, pathlib, collections, sys, math, statistics, re
import pandas as pd
import ir_measures as irm
from datasets import load_dataset
from ir_measures import Qrel, ScoredDoc

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy is optional at runtime
    _scipy_stats = None

# ---------- loaders (fault-tolerant) ----------
def _load_qrels_local(path: pathlib.Path):
    """
    Accepts:
      1) BEIR TSV, with or without header:
         query-id <tab> corpus-id <tab> score
      2) Classic TREC whitespace format:
         qid 0 docid rel
    Ignores comment lines starting with '#'.
    """
    def adapt(parts):
        """Return (qid, docid, rel) from split line."""
        if len(parts) == 3:                     # BEIR
            qid, docid, rel = parts
        elif len(parts) == 4:                   # TREC
            qid, _zero, docid, rel = parts
        else:
            raise ValueError(f"Unrecognised qrels line: {' '.join(parts)}")
        return Qrel(qid, docid, int(rel))

    qrels = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue                        # skip blank / comment lines
            parts = line.split()                # works for space *or* tab
            # skip a header row like: query-id  corpus-id  score
            if parts[0].lower() in {"query-id", "qid"} and not parts[0].isdigit():
                continue
            qrels.append(adapt(parts))
    return qrels

# ---------- loader for HF-Hub qrels (streaming) ----------
### NEW ###
def _load_qrels_hf(repo, config, split, id_field, doc_field, rel_field,
                   trust_remote, streaming=True):
    ds = load_dataset(repo, config, split=split,
                      streaming=streaming, trust_remote_code=trust_remote)
    return [Qrel(str(r[id_field]), str(r[doc_field]), int(r[rel_field]))
            for r in ds]

def load_run(path: pathlib.Path):
    return [ScoredDoc(q, d, float(s))
            for q, _, d, _, s, _ in map(str.split, path.open())]

# ---------- defaults ----------
_DEFAULT_METRICS = """
 nDCG@1 nDCG@3 nDCG@5 nDCG@10 nDCG@100
 MRR@1 MRR@3 MRR@5 MRR@10 MRR@100
 P@1  P@3  P@5  P@10  P@100
 Recall@1 Recall@3 Recall@5 Recall@10 Recall@100
 AP MAP
""".split()

_PERCENT_SCALE = 100.0

def _parse_metric_tokens(tokens):
    if hasattr(irm, "parse"):                                # ≥ 0.4
        return list(irm.parse(" ".join(tokens)))
    return [irm.parse_measure(t) for t in tokens]            # ≤ 0.3.7


def _scale_metric_value(value):
    if value is None:
        return None
    try:
        return float(value) * _PERCENT_SCALE
    except (TypeError, ValueError):
        return value


def _resolve_qrels_cache_path(cache_path, repo, config, split):
    path = pathlib.Path(cache_path)
    if path.exists() and path.is_dir():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{repo}-{config}-{split}")
        return path / f"{safe}.tsv"
    if path.suffix:
        return path
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{repo}-{config}-{split}")
    return path / f"{safe}.tsv"


def _qrel_relevance(qrel):
    if hasattr(qrel, "relevance"):
        return qrel.relevance
    if hasattr(qrel, "rel"):
        return qrel.rel
    try:
        return qrel[2]
    except Exception:
        return None


def _write_qrels_tsv(path, qrels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for q in qrels:
            rel = _qrel_relevance(q)
            if rel is None:
                continue
            fh.write(f"{q.query_id}\t{q.doc_id}\t{rel}\n")
    tmp_path.replace(path)


def _t_multiplier(conf_level, n):
    """Return the two-tailed t critical value for confidence level and sample size."""
    if n < 2:
        return math.nan
    prob = 1 - (1 - conf_level) / 2
    if _scipy_stats:
        return float(_scipy_stats.t.ppf(prob, df=n - 1))
    try:
        return statistics.NormalDist().inv_cdf(prob)         # normal fallback
    except Exception:
        return math.nan


def _ndcg_statistics(values, mean):
    """Compute std, stderr and confidence intervals for a list of nDCG values."""
    n = len(values)
    if n == 0:
        return None

    std = statistics.stdev(values) if n > 1 else 0.0
    se  = std / math.sqrt(n) if n else math.nan

    def _ci(level):
        t_mult = _t_multiplier(level, n)
        if math.isnan(t_mult) or math.isnan(se):
            return (math.nan, math.nan)
        delta = t_mult * se
        return (mean - delta, mean + delta)

    ci90_low, ci90_high = _ci(0.90)
    ci95_low, ci95_high = _ci(0.95)

    return {
        "std": std,
        "stderr": se,
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run",     required=True)
    ap.add_argument("--split",   default="dev")
    ap.add_argument("--outdir",  default="results")
    ap.add_argument("--qrels")

     # ▼ NEW: load qrels from the Hub
    ap.add_argument("--qrels_repo",   help="HF repo, e.g. BeIR/msmarco-qrels")
    ap.add_argument("--qrels_config", default="default",
                    help="HF config name (dataset-specific)")
    ap.add_argument("--qrels_split",  default="validation")
    ap.add_argument("--id_field",     default="query-id",
                    help="column with query ids in the qrels dataset")
    ap.add_argument("--doc_field",    default="corpus-id",
                    help="column with doc ids in the qrels dataset")
    ap.add_argument("--rel_field",    default="score",
                    help="column with relevance      in the qrels dataset")
    ap.add_argument("--trust_remote", action="store_true",
                    help="Pass trust_remote_code=True to load_dataset")
    ap.add_argument("--qrels_cache",
                    help="Cache path for HF qrels TSV; if directory, "
                         "file name is derived from repo/config/split")
    ap.add_argument("--qrels_streaming", action="store_true",
                    help="Stream qrels from HF (disables cache write)")

    ap.add_argument("--metrics", nargs="+")
    ap.add_argument("--perquery", action="store_true")
    ap.add_argument("--filter_docids",
                    help="File with doc-ids actually indexed; "
                         "drops qrels that refer to missing docs")
    ap.add_argument("--filter_qids",
                help="Path to TSV/CSV/TXT with common qids; "
                     "keeps only these queries in both qrels and run.")
    
    args = ap.parse_args()

    cache_path = None
    if args.qrels_repo:                                    # ← NEW branch
        if args.qrels_cache and not args.qrels_streaming:
            cache_path = _resolve_qrels_cache_path(
                args.qrels_cache,
                args.qrels_repo,
                args.qrels_config,
                args.qrels_split,
            )
            if cache_path.exists():
                print(f"[i] Using cached qrels from {cache_path}")
                qrels = _load_qrels_local(cache_path)
            else:
                print(f"[i] Downloading qrels from {args.qrels_repo}/{args.qrels_config} "
                      f"split={args.qrels_split}")
                qrels = _load_qrels_hf(args.qrels_repo, args.qrels_config,
                                       args.qrels_split,
                                       args.id_field, args.doc_field, args.rel_field,
                                       args.trust_remote,
                                       streaming=False)
                _write_qrels_tsv(cache_path, qrels)
                print(f"[i] Cached qrels to {cache_path}")
        else:
            print(f"[i] Streaming qrels from {args.qrels_repo}/{args.qrels_config} "
                  f"split={args.qrels_split}")
            qrels = _load_qrels_hf(args.qrels_repo, args.qrels_config,
                                   args.qrels_split,
                                   args.id_field, args.doc_field, args.rel_field,
                                   args.trust_remote,
                                   streaming=True)
    else:
        if args.qrels:
            qrels_path = pathlib.Path(args.qrels)
        else:
            base = os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data"))
            droot = pathlib.Path(base) / args.dataset
            qrels_path = next((droot / "qrels").glob(f"{args.split}.*"), None)
            if not qrels_path:
                sys.exit(f"[ERROR] No qrels for split {args.split} under {droot/'qrels'}")
        qrels = _load_qrels_local(qrels_path)

    # --- sign-post: header ---------------
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*72)
    print(f"[{start_time}]  BEGIN  dataset={args.dataset}  split={args.split}")
    print(f"Run file : {args.run}")
    if args.qrels_repo:
        print(f"Qrels repo: {args.qrels_repo}/{args.qrels_config} "
              f"split={args.qrels_split}")
        if cache_path:
            print(f"Qrels cache: {cache_path}")
    else:
        print(f"Qrels    : {qrels_path}")
    print("="*72 + "\n")

    # --- load data -----------------------
    # qrels = load_qrels(qrels_path)
    run   = load_run(pathlib.Path(args.run))

    # --- optional filtering (subset evaluation) ------------------------
    if args.filter_docids:
        allowed = set(pathlib.Path(args.filter_docids).read_text().splitlines())
        qrels = [q for q in qrels if q.doc_id in allowed]
        print(f"[i] Filtered qrels: {len(allowed)} docs kept.")

    if not qrels:
        sys.exit("[ERROR] After filtering, no qrels remain; "
             "metrics would all be zero. "
             "Check --filter_docids or your corpus subset.")
        
    # --- optional filtering (subset evaluation) ------------------------
    if args.filter_qids:
        path = pathlib.Path(args.filter_qids)
        lines = path.read_text(encoding="utf-8").splitlines()
        allowed = {line.strip().split()[0] for line in lines if line.strip()}

        # keep only qids in common set
        qrels = [q for q in qrels if str(q.query_id) in allowed]
        run   = [d for d in run   if str(d.query_id) in allowed]

        kept_qs = len({q.query_id for q in qrels})
        print(f"[i] Filtered to {kept_qs} queries by --filter_qids from {path}")


    metric_tokens = args.metrics if args.metrics else _DEFAULT_METRICS
    measures      = _parse_metric_tokens(metric_tokens)
    metric_names  = [str(m) for m in measures]

    ndcg_values = collections.defaultdict(list)
    perquery_rows = []
    for qid, metric, value in irm.iter_calc(measures, qrels, run):
        metric_str = str(metric)
        scaled_value = _scale_metric_value(value)
        if metric_str.lower().startswith("ndcg") and scaled_value is not None:
            try:
                val = float(scaled_value)
            except (TypeError, ValueError):
                val = math.nan
            if not math.isnan(val):
                ndcg_values[metric_str].append(val)
        if args.perquery:
            perquery_rows.append((qid, metric_str, scaled_value))

    # --- aggregate -----------------------
    agg = irm.calc_aggregate(measures, qrels, run)
    ordered = collections.OrderedDict()
    for m in measures:
        mname = str(m)
        scaled_value = _scale_metric_value(agg[m])
        ordered[mname] = scaled_value
        if mname.lower().startswith("ndcg") and scaled_value is not None:
            stats = _ndcg_statistics(ndcg_values.get(mname, []), scaled_value)
            if stats:
                ordered[f"{mname}_std"]       = stats["std"]
                ordered[f"{mname}_stderr"]    = stats["stderr"]
                ordered[f"{mname}_ci90_low"]  = stats["ci90_low"]
                ordered[f"{mname}_ci90_high"] = stats["ci90_high"]
                ordered[f"{mname}_ci95_low"]  = stats["ci95_low"]
                ordered[f"{mname}_ci95_high"] = stats["ci95_high"]

    print(pd.Series(ordered).to_string(float_format="%.4f"), "\n")

    # --- per-query (optional) ------------
    if args.perquery:
        perq = (pd.DataFrame(perquery_rows, columns=["qid", "metric", "val"])
                  .pivot(index="qid", columns="metric", values="val")
                  .reindex(columns=metric_names)
                  .reset_index())

    # --- diagnostics ---------------------
    qrels_docs = {(q.query_id, q.doc_id) for q in qrels}
    retrieved  = {(d.query_id, d.doc_id) for d in run}
    overlap    = qrels_docs & retrieved
    diag = {
        "num_queries" : len({q.query_id for q in qrels}),
        "num_retrieved": len(retrieved),
        "num_relevant" : len(qrels_docs),
        "num_overlap"  : len(overlap),
        "pct_unjudged" : (1 - len(overlap)/max(1, len(retrieved))) * _PERCENT_SCALE
    }

    # --- save ----------------------------
    ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = pathlib.Path(args.run).stem
    base = f"{stem}_{args.split}_{ts}"
    out  = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([ordered]).to_csv(out/f"{base}-agg.csv", index=False)
    json.dump(ordered, open(out/f"{base}-agg.json", "w"), indent=2)
    if args.perquery:
        perq.to_csv(out/f"{base}-perquery.csv", index=False)
    with open(out/f"{base}-diagnostic.txt", "w") as fh:
        for k, v in diag.items(): print(f"{k}: {v}", file=fh)

    # --- sign-post: footer ---------------
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("="*72)
    print(f"[{end_time}]  END    dataset={args.dataset}  split={args.split}")
    print("="*72 + "\n")

if __name__ == "__main__":
    main()

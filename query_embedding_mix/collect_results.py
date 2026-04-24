#!/usr/bin/env python3
"""
Collect IR results from an existing folder into one CSV, plus a processed summary.

Expected experiment root folder names:
  <dataset>-<docs_size>-<doc_lang>-<query_lang_1>-<query_lang_2>-<num_bands>-<model>
e.g., mmarco-100k-english-zh-en-5-bge-m3

Output schema:
  pair, doc_mix, method, mix_ratio, ndcg10, ndcg10_std, ndcg10_stderr,
  ndcg10_ci90_low, ndcg10_ci90_high, ndcg10_ci95_low, ndcg10_ci95_high,
  mrr10, r10, model, source_file

Notes:
- word-mix band labels like "20-40" -> mix_ratio midpoint (30) by default.
- embed-mix ratios 0..1 -> convert to percent 0..100.
- CSVs:
  * If they already have "metric,mean" columns, we use them.
  * If they have per-query columns (e.g., ndcg@10, mrr@10), we compute the column mean.
"""

import os
import re
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Union, List
import pandas as pd
import argparse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "mmarco_full"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "tables" / "full_mmarco_results.csv"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "artifacts" / "tables" / "full_mmarco_processed_results.csv"
DEFAULT_TYPOLOGY_METRICS_PATH = PROJECT_ROOT / "configs" / "language_pairs_typology_metrics.csv"

WORD_BAND_STRATEGY = "midpoint"  # change to "band_label" to keep "20-40" as text

LANG_MAP = {
    "amharic": "AM", "am": "AM",
    "english": "EN", "en": "EN",
    "chinese": "ZH", "zh": "ZH", "cn": "ZH",
    "khmer": "KM", "km": "KM",
    "kurdish": "KU", "ku": "KU",
    "burmese": "MY", "myanmar": "MY", "my": "MY",
    "swahili": "SW", "sw": "SW",
    "shan": "SHN", "shn": "SHN",
    "slovene": "SL", "slovenian": "SL", "solvene": "SL", "sl": "SL",
    "nepali": "NE", "ne": "NE",
    "sinhala": "SI", "si": "SI",
    "indonesian": "ID", "indo": "ID", "id": "ID",
    "arabic": "AR", "ar": "AR",
    "german": "DE", "de": "DE",
    "spanish": "ES", "es": "ES",
    "french": "FR", "fr": "FR",
    "hindi": "HI", "hi": "HI",
    "italian": "IT", "it": "IT",
    "japanese": "JA", "ja": "JA",
    "dutch": "NL", "nl": "NL",
    "portuguese": "PT", "pt": "PT",
    "russian": "RU", "ru": "RU",
    "vietnamese": "VI", "vi": "VI",
}
DOC_LABELS = {code: f"{code} docs" for code in sorted(set(LANG_MAP.values()))}
LANG_TOKEN_SET = {k.lower() for k in LANG_MAP.keys()}

# Language factors derived from language_summary.md
LANG_INFO: Dict[str, Dict[str, str]] = {
    "ar": {"script": "arabic", "family": "afro-asiatic/sem", "typology": "templatic_vso/svo", "resource": "5"},
    "de": {"script": "latin", "family": "indo-european/germanic", "typology": "fusional_v2", "resource": "5"},
    "en": {"script": "latin", "family": "indo-european/germanic", "typology": "analytic_svo", "resource": "5"},
    "es": {"script": "latin", "family": "indo-european/romance", "typology": "fusional_svo", "resource": "5"},
    "fr": {"script": "latin", "family": "indo-european/romance", "typology": "fusional_svo", "resource": "5"},
    "hi": {"script": "devanagari", "family": "indo-european/indo-aryan", "typology": "fusional_agglutinative_sov", "resource": "4"},
    "id": {"script": "latin", "family": "austronesian", "typology": "analytic_svo", "resource": "3"},
    "it": {"script": "latin", "family": "indo-european/romance", "typology": "fusional_svo", "resource": "4"},
    "ja": {"script": "kanji-kana", "family": "japonic", "typology": "agglutinative_sov", "resource": "5"},
    "nl": {"script": "latin", "family": "indo-european/germanic", "typology": "fusional_v2", "resource": "4"},
    "pt": {"script": "latin", "family": "indo-european/romance", "typology": "fusional_svo", "resource": "4"},
    "ru": {"script": "cyrillic", "family": "indo-european/slavic", "typology": "fusional_svo", "resource": "4"},
    "vi": {"script": "latin", "family": "austroasiatic/vietic", "typology": "analytic_svo", "resource": "4"},
    "zh": {"script": "han", "family": "sino-tibetan/sinitic", "typology": "analytic_svo", "resource": "5"},
}
METRICS_TO_KEEP = [
    "ndcg@10",
    "ndcg@10_std",
    "ndcg@10_stderr",
    "ndcg@10_ci90_low",
    "ndcg@10_ci90_high",
    "ndcg@10_ci95_low",
    "ndcg@10_ci95_high",
    "rr@10",
    "r@10",
]  # e.g., ["ndcg@10","mrr@10"]; keep all when set to None.
METRICS_TO_KEEP_SET = (
    {m.lower() for m in METRICS_TO_KEEP} if METRICS_TO_KEEP is not None else None
)
METRIC_EXPORT_MAP = {
    "ndcg@10": "ndcg10",
    "ndcg@10_std": "ndcg10_std",
    "ndcg@10_stderr": "ndcg10_stderr",
    "ndcg@10_ci90_low": "ndcg10_ci90_low",
    "ndcg@10_ci90_high": "ndcg10_ci90_high",
    "ndcg@10_ci95_low": "ndcg10_ci95_low",
    "ndcg@10_ci95_high": "ndcg10_ci95_high",
    "mrr@10": "mrr10",
    "rr@10": "mrr10",  # RR@10 is equivalent to MRR@10 for our runs.
    "r@10": "r10",
}
METRIC_SCALE = 1.0  # evaluate.py already outputs metrics in 0-100 scale.
DELTA_BOOTSTRAP_ITER = 10000
DELTA_BOOTSTRAP_SEED = 42
EXPORT_COLUMNS = [
    "pair",
    "doc_mix",
    "method",
    "mix_ratio",
    "ndcg10",
    "ndcg10_std",
    "ndcg10_stderr",
    "ndcg10_ci90_low",
    "ndcg10_ci90_high",
    "ndcg10_ci95_low",
    "ndcg10_ci95_high",
    "mrr10",
    "r10",
    "model",
    "source_file",
]

TIMESTAMP_RE = re.compile(r"(\d{8})[-_](\d{6})")
# Matches strings like:
#   *_dev_20251231-175309*
#   *_dev_20251231_175309-2802513*
# and strips the whole "phase + timestamp (+ optional job id)" block.
PHASE_TIMESTAMP_RE = re.compile(
    r"_(?:dev|test|validation|val|train)[-_]\d{8}[-_]\d{6}(?:[-_]\d+)?",
    re.IGNORECASE,
)
BARE_TIMESTAMP_RE = re.compile(r"[-_]\d{8}[-_]\d{6}(?:[-_]\d+)?")

def human_doc_mix(doc_lang_token: str, pair: Optional[str] = None) -> str:
    toks = re.split(r"[_\-&+]", (doc_lang_token or "").lower())
    codes = sorted({LANG_MAP.get(t.strip(), t.strip().upper()) for t in toks if t.strip()})
    codes = [c for c in codes if c in DOC_LABELS]
    if not codes:
        if doc_lang_token and "bilingual" in doc_lang_token.lower() and pair:
            pp = [p.strip().upper() for p in pair.split("-") if p.strip()]
            if len(pp) == 2:
                return f"{pp[0]} + {pp[1]} docs"
        return doc_lang_token or "docs"
    if len(codes) == 1: return DOC_LABELS.get(codes[0], f"{codes[0]} docs")
    return " & ".join(codes) + " docs"

def pair_from_tokens(q1: Optional[str], q2: Optional[str]) -> str:
    a = LANG_MAP.get((q1 or "EN").lower(), (q1 or "EN").upper())
    b = LANG_MAP.get((q2 or "ZH").lower(), (q2 or "ZH").upper())
    return f"{a}-{b}"

def midpoint_from_band(band: str) -> Optional[float]:
    band = band.strip()
    if band in {"0","100"}: return float(band)
    m = re.match(r"^\s*(\d+)\s*[-_]\s*(\d+)\s*$", band)
    if not m: return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo + hi) / 2

def infer_method_and_ratio_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer ('word' or 'embed', ratio label) from folder/filename."""
    parts = [p.lower() for p in path.parts]
    method = None
    ratio_label = None
    if any("vector_mix" in p or "vectormix" in p for p in parts):
        method = "embed"
    if any(("word" in p and "mix" in p) or ("word_mix" in p) or ("wordmix" in p) for p in parts):
        method = "word"

    name = path.stem.lower()
    norm_name = re.sub(r"[_\s]+", "-", name)

    m = re.search(r"(?<!\d)(\d{1,3})\s*[-_]\s*(\d{1,3})(?!\d)", norm_name)
    if m:
        ratio_label = f"{m.group(1)}-{m.group(2)}"
    if ratio_label is None:
        m = re.search(r"(?:alpha|ratio|mix|band|cm|wm)[-_]?(\d+(?:\.\d+)?)", norm_name)
        if m:
            ratio_label = m.group(1)
    if ratio_label is None:
        if re.search(r"(?<![\d.])100(?![\d.])", norm_name):
            ratio_label = "100"
        elif re.search(r"(?<![\d.])0(?![\d.])", norm_name):
            ratio_label = "0"
    if ratio_label is None:
        for match in re.finditer(r"(\d+(?:\.\d+)?)", norm_name):
            num = match.group(1)
            try:
                val = float(num)
            except ValueError:
                continue
            if val <= 100:
                ratio_label = num
                break
    return method, ratio_label

def parse_folder_name(name: str):
    parts = name.split("-")
    info = {"dataset":None,"docs_size":None,"doc_lang":None,"q1":None,"q2":None,"num_bands":None,"model":None}
    if not parts:
        return info
    info["dataset"] = parts[0]
    if len(parts) >= 2:
        info["docs_size"] = parts[1]
    idx = 2
    if idx < len(parts):
        info["doc_lang"] = parts[idx]
        idx += 1

    langs: List[str] = []
    while idx < len(parts) and len(langs) < 2:
        token = parts[idx].lower()
        if token in LANG_TOKEN_SET:
            langs.append(parts[idx])
            idx += 1
        else:
            break
    if langs:
        info["q1"] = langs[0]
        if len(langs) > 1:
            info["q2"] = langs[1]

    if idx < len(parts) and "band" in parts[idx].lower():
        info["num_bands"] = parts[idx]
        idx += 1

    if idx < len(parts):
        info["model"] = "-".join(parts[idx:])
    return info

def find_experiment_root(path: Path):
    """Walk upwards until we find a folder name that has q1 & q2."""
    for p in [path] + list(path.parents):
        info = parse_folder_name(p.name)
        if info.get("q1") and info.get("q2"):
            return info
    return parse_folder_name(path.name)

def normalize_pair(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return (
        re.sub(r"\s+", "", value.strip())
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .upper()
    )

def normalize_doc_mix(value: str) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def parse_resource_level(value: str) -> float:
    if not isinstance(value, str):
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else float("nan")

def resource_class(level: float) -> str:
    if math.isnan(level):
        return "U"
    return "H" if level >= 5 else "L"

def split_pair_codes(pair: str) -> Tuple[str, str]:
    cleaned = normalize_pair(pair)
    parts = [p for p in re.split(r"[-/]", cleaned) if p]
    if len(parts) >= 2:
        return parts[0].lower(), parts[1].lower()
    if len(parts) == 1:
        return parts[0].lower(), parts[0].lower()
    return ("", "")


def canonical_pair_key(pair: str) -> str:
    a, b = split_pair_codes(pair)
    if a and b:
        return "-".join(sorted((a.upper(), b.upper())))
    return normalize_pair(pair)


def load_pair_typology_metrics(csv_path: Path = DEFAULT_TYPOLOGY_METRICS_PATH) -> Dict[str, Dict[str, float]]:
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required_cols = {"src_lang", "tgt_lang"}
    if not required_cols.issubset(df.columns):
        return {}

    metric_map = {
        "lang2vec_knn": "lang2vec_knn",
        "grambank": "gram_bank",
        "scripts": "script",
        "glot_tree": "glot_tree",
    }
    out: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        src = LANG_MAP.get(str(row["src_lang"]).strip().lower(), str(row["src_lang"]).strip().upper())
        tgt = LANG_MAP.get(str(row["tgt_lang"]).strip().lower(), str(row["tgt_lang"]).strip().upper())
        if not src or not tgt:
            continue

        values: Dict[str, float] = {}
        for csv_col, out_col in metric_map.items():
            if csv_col not in df.columns:
                continue
            value = pd.to_numeric(pd.Series([row[csv_col]]), errors="coerce").iloc[0]
            if pd.notna(value):
                values[out_col] = float(value)

        if values:
            out[canonical_pair_key(f"{src}-{tgt}")] = values

    return out


PAIR_EXTRA_METRICS = load_pair_typology_metrics()

def pair_factors(pair: str) -> Dict[str, Union[str, float]]:
    a, b = split_pair_codes(pair)
    info_a = LANG_INFO.get(a, {})
    info_b = LANG_INFO.get(b, {})
    script_match = "match" if info_a.get("script") == info_b.get("script") and info_a else "mismatch"
    family_dist = 0 if info_a.get("family") == info_b.get("family") and info_a else 1
    typology_dist = 0 if info_a.get("typology") == info_b.get("typology") and info_a else 1
    res_a = parse_resource_level(info_a.get("resource", "")) if info_a else float("nan")
    res_b = parse_resource_level(info_b.get("resource", "")) if info_b else float("nan")
    res_pattern = f"{resource_class(res_a)}-{resource_class(res_b)}"
    return {
        "lang_a": a,
        "lang_b": b,
        "script_match": script_match,
        "family_dist": family_dist,
        "typology_dist": typology_dist,
        "resource_pattern": res_pattern,
    }

def add_doc_type(doc_mix: str) -> str:
    if " + " in doc_mix or "+" in doc_mix:
        return "bi"
    return "mono"

def infer_doc_regime(doc_mix: str, pair: str) -> str:
    la, lb = split_pair_codes(pair)
    dm = normalize_doc_mix(doc_mix).upper()
    has_a = bool(la) and re.search(rf"\b{re.escape(la.upper())}\b", dm) is not None
    has_b = bool(lb) and re.search(rf"\b{re.escape(lb.upper())}\b", dm) is not None
    if has_a and has_b:
        return "L1+L2 docs"
    if has_a:
        return "L1 docs"
    if has_b:
        return "L2 docs"
    return "other docs"

def load_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    for col in ("pair", "doc_mix", "mix_ratio", "ndcg10"):
        if col not in out.columns:
            out[col] = ""
    out["pair"] = out["pair"].apply(normalize_pair)
    out["doc_mix"] = out["doc_mix"].apply(normalize_doc_mix)
    out["mix_ratio"] = pd.to_numeric(out["mix_ratio"], errors="coerce")
    out["ndcg10"] = pd.to_numeric(out["ndcg10"], errors="coerce")
    return out

def compute_full_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (pair, doc_mix), grp in df.groupby(["pair", "doc_mix"]):
        grp = grp.sort_values(
            [col for col in ["mix_ratio", "method", "model", "source_file", "result_id"] if col in grp.columns],
            kind="mergesort",
        ).reset_index(drop=True)
        endpoints = grp[grp["mix_ratio"].isin([0, 100])]
        midpoints = grp[(grp["mix_ratio"] > 0) & (grp["mix_ratio"] < 100)]
        best_endpoint_ndcg = endpoints["ndcg10"].max() if not endpoints.empty else float("nan")
        if midpoints.empty:
            best_mixed_ndcg = float("nan")
            delta_ndcg = 0.0
            lambda_star_mid = float("nan")
        else:
            best_mid_sort_cols = [
                col for col in ["ndcg10", "mix_ratio", "method", "model", "source_file", "result_id"]
                if col in midpoints.columns
            ]
            best_mid_ascending = [False] + [True] * (len(best_mid_sort_cols) - 1)
            best_mid_row = midpoints.sort_values(
                best_mid_sort_cols,
                ascending=best_mid_ascending,
                kind="mergesort",
            ).iloc[0]
            best_mixed_ndcg = float(best_mid_row["ndcg10"])
            lambda_star_mid = float(best_mid_row["mix_ratio"])
            delta_ndcg = best_mixed_ndcg - (best_endpoint_ndcg if not math.isnan(best_endpoint_ndcg) else 0.0)
        perquery_items: List[Tuple[float, pd.Series]] = []
        if "perquery_ndcg10" in grp.columns:
            for _, row in grp.iterrows():
                series = row.get("perquery_ndcg10")
                if isinstance(series, pd.Series):
                    ratio = pd.to_numeric(row.get("mix_ratio"), errors="coerce")
                    if pd.notna(ratio):
                        perquery_items.append((float(ratio), series))
        delta_ci = _bootstrap_delta_ndcg_ci(perquery_items)
        row = {
            "pair": pair,
            "doc_mix": doc_mix,
            "best_endpoint_ndcg": best_endpoint_ndcg,
            "best_mixed_ndcg": best_mixed_ndcg,
            "delta_ndcg": delta_ndcg,
            "lambda_star_mid": lambda_star_mid,
            "delta_ndcg_ci90_low": delta_ci["delta_ndcg_ci90_low"] if delta_ci else float("nan"),
            "delta_ndcg_ci90_high": delta_ci["delta_ndcg_ci90_high"] if delta_ci else float("nan"),
            "delta_ndcg_ci95_low": delta_ci["delta_ndcg_ci95_low"] if delta_ci else float("nan"),
            "delta_ndcg_ci95_high": delta_ci["delta_ndcg_ci95_high"] if delta_ci else float("nan"),
        }
        row.update(pair_factors(str(pair)))
        row.update(PAIR_EXTRA_METRICS.get(canonical_pair_key(str(pair)), {}))
        row["doc_type"] = add_doc_type(str(doc_mix))
        row["doc_regime"] = infer_doc_regime(str(doc_mix), str(pair))
        rows.append(row)
    return pd.DataFrame(rows)

def to_mix_ratio_value(method: str, ratio_label: Optional[str]) -> Optional[Union[float, str]]:
    if ratio_label is None: return None
    r = ratio_label.strip()
    if method == "word":
        if WORD_BAND_STRATEGY == "band_label": return r
        if re.match(r"^\d+[-_]\d+$", r):
            mp = midpoint_from_band(r.replace("_","-"))
            return mp if mp is not None else r
        try:
            val = float(r)
            return round(val*100, 4) if 0 <= val <= 1 else val
        except ValueError:
            return r
    # embed: normalize 0..1 -> 0..100
    try:
        val = float(r)
        return round(val*100, 4) if 0 <= val <= 1.0 else float(val)
    except:
        return r

def _canonical_result_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = PHASE_TIMESTAMP_RE.sub("", stem)
    stem = BARE_TIMESTAMP_RE.sub("", stem)
    stem = re.sub(r"__+", "_", stem)
    stem = re.sub(r"--+", "-", stem)
    return stem.strip("_-")

def _result_timestamp(path: Path) -> float:
    name = path.stem
    m = TIMESTAMP_RE.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0

def _is_perquery_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("-perquery") or stem.endswith("_perquery")

def _base_result_key(path: Path) -> str:
    key = _canonical_result_key(path)
    for suffix in ("-agg", "-perquery", "_agg", "_perquery"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key

def _result_id(root_dir: Path, path: Path) -> str:
    try:
        rel = path.parent.relative_to(root_dir)
    except ValueError:
        rel = path.parent
    return f"{rel.as_posix()}/{_base_result_key(path)}"

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        col = lower.get(name.lower())
        if col is not None:
            return col
    return None

def _extract_perquery_ndcg10(df: pd.DataFrame) -> Optional[pd.Series]:
    qid_col = _find_column(df, ["qid", "query_id", "query-id", "query", "topic"])
    metric_col = _find_column(df, ["ndcg@10"])
    if not qid_col or not metric_col:
        return None
    qids = df[qid_col].astype(str)
    vals = pd.to_numeric(df[metric_col], errors="coerce")
    mask = vals.notna()
    if not mask.any():
        return None
    series = pd.Series(vals[mask].values, index=qids[mask].values)
    if METRIC_SCALE != 1.0:
        series = series * METRIC_SCALE
    return series.groupby(level=0).mean()

def _is_endpoint_ratio(ratio: float) -> bool:
    return math.isclose(ratio, 0.0) or math.isclose(ratio, 100.0)


def _stable_qid_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    if re.fullmatch(r"[+-]?\d+", text):
        return (0, int(text))
    return (1, text)


def _bootstrap_delta_ndcg_ci(
    items: List[Tuple[float, pd.Series]],
    iterations: int = DELTA_BOOTSTRAP_ITER,
    seed: int = DELTA_BOOTSTRAP_SEED,
) -> Optional[Dict[str, float]]:
    if iterations < 2 or not items:
        return None
    common_qids = None
    for _, series in items:
        idx = set(series.index)
        common_qids = idx if common_qids is None else common_qids & idx
    if not common_qids or len(common_qids) < 2:
        return None
    qids = sorted(common_qids, key=_stable_qid_sort_key)
    ratios: List[float] = []
    arrays: List[List[float]] = []
    for ratio, series in sorted(items, key=lambda item: float(item[0])):
        vals = pd.to_numeric(series.loc[qids], errors="coerce").to_numpy(dtype=float)
        if vals.size == 0 or (np is not None and np.all(np.isnan(vals))):
            continue
        ratios.append(float(ratio))
        arrays.append(vals)
    if not arrays:
        return None
    mid_idx = [i for i, r in enumerate(ratios) if 0.0 < r < 100.0]
    end_idx = [i for i, r in enumerate(ratios) if _is_endpoint_ratio(r)]
    if not mid_idx or not end_idx:
        return None
    n = arrays[0].shape[0]
    if np is None:
        rng = random.Random(seed)
        deltas: List[float] = []
        for _ in range(iterations):
            sample_idx = [rng.randrange(n) for _ in range(n)]
            best_mid = float("-inf")
            best_end = float("-inf")
            for idx, arr in enumerate(arrays):
                vals = [arr[i] for i in sample_idx if not math.isnan(arr[i])]
                mean_val = sum(vals) / len(vals) if vals else float("nan")
                if math.isnan(mean_val):
                    continue
                if idx in end_idx:
                    if mean_val > best_end:
                        best_end = mean_val
                elif idx in mid_idx:
                    if mean_val > best_mid:
                        best_mid = mean_val
            if best_mid == float("-inf") or best_end == float("-inf"):
                continue
            deltas.append(best_mid - best_end)
        if not deltas:
            return None
        deltas_series = pd.Series(deltas)
        return {
            "delta_ndcg_ci90_low": float(deltas_series.quantile(0.05)),
            "delta_ndcg_ci90_high": float(deltas_series.quantile(0.95)),
            "delta_ndcg_ci95_low": float(deltas_series.quantile(0.025)),
            "delta_ndcg_ci95_high": float(deltas_series.quantile(0.975)),
        }
    stack = np.vstack(arrays)  # shape: (ratios, queries)
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, n, size=(iterations, n))
    means = np.nanmean(stack[:, idxs], axis=2)  # shape: (ratios, iterations)
    best_mid = np.nanmax(means[mid_idx, :], axis=0)
    best_end = np.nanmax(means[end_idx, :], axis=0)
    deltas = best_mid - best_end
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return None
    ci90_low, ci90_high = np.quantile(deltas, [0.05, 0.95])
    ci95_low, ci95_high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_ndcg_ci90_low": float(ci90_low),
        "delta_ndcg_ci90_high": float(ci90_high),
        "delta_ndcg_ci95_low": float(ci95_low),
        "delta_ndcg_ci95_high": float(ci95_high),
    }

def select_latest_csv_files(dirpath: Path, filenames: List[str]) -> List[Path]:
    latest: Dict[str, Tuple[Path, float]] = {}
    for fn in sorted(filenames):
        if not fn.lower().endswith(".csv"):
            continue
        fp = dirpath / fn
        key = _canonical_result_key(fp)
        ts = _result_timestamp(fp)
        existing = latest.get(key)
        if (
            existing is None
            or ts > existing[1]
            or (ts == existing[1] and fp.as_posix() < existing[0].as_posix())
        ):
            latest[key] = (fp, ts)
    return [latest[key][0] for key in sorted(latest)]

def extract_means_from_csv(df: pd.DataFrame) -> Dict[str, float]:
    """Use 'metric,mean' if present; else compute means from metric-like columns."""
    out: Dict[str, float] = {}
    lower = {c.lower(): c for c in df.columns}
    if "metric" in lower and "mean" in lower:
        mcol, vcol = lower["metric"], lower["mean"]
        for metric, sub in df.groupby(mcol):
            try:
                val = float(sub[vcol].iloc[0])
            except Exception:
                val = pd.to_numeric(sub[vcol], errors="coerce").mean()
            if pd.notna(val):
                out[str(metric).lower()] = float(val)
        return out
    # Try per-query metric columns
    candidates = []
    for c in df.columns:
        lower_name = c.lower()
        if lower_name in {"qid","query","query_id","topic","docid","docno","question"}:
            continue
        if lower_name.startswith("unnamed"):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            candidates.append((c, vals))
    for c, vals in candidates:
        metric_name = c.lower()
        out[metric_name] = float(vals.mean())
    if METRICS_TO_KEEP_SET is not None:
        out = {k: v for k, v in out.items() if k in METRICS_TO_KEEP_SET}
    return out

def collect_results(root_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    perquery_map: Dict[str, pd.Series] = {}
    root_info = parse_folder_name(root_dir.name)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        filenames = sorted(filenames)
        dpath = Path(dirpath)
        if dpath == root_dir and not (root_info.get("q1") and root_info.get("q2")):
            dirnames[:] = [
                dname for dname in dirnames
                if parse_folder_name(dname).get("q1") and parse_folder_name(dname).get("q2")
            ]
        folder = find_experiment_root(dpath)
        if not folder.get("q1") or not folder.get("q2"):
            continue
        latest_files = select_latest_csv_files(dpath, filenames)
        for fp in latest_files:
            result_id = _result_id(root_dir, fp)
            is_perquery = _is_perquery_file(fp)

            method, ratio_label = infer_method_and_ratio_from_path(fp)
            if method is None:
                method = "embed" if "vector_mix" in dpath.as_posix().lower() else "word"
            mix_ratio = to_mix_ratio_value(method, ratio_label)

            try:
                df = pd.read_csv(fp)
            except Exception:
                continue

            if is_perquery:
                perquery_series = _extract_perquery_ndcg10(df)
                if perquery_series is not None:
                    perquery_map[result_id] = perquery_series
                continue

            pair = pair_from_tokens(folder.get("q1"), folder.get("q2"))
            doc_mix = human_doc_mix(folder.get("doc_lang") or "docs", pair=pair)

            metrics = extract_means_from_csv(df)
            if not metrics:
                continue

            row = {
                "result_id": result_id,
                "pair": pair,
                "doc_mix": doc_mix,
                "method": method,
                "mix_ratio": mix_ratio,
                "model": folder.get("model"),
                "source_file": str(fp.relative_to(root_dir)),
                "ndcg10": None,
                "ndcg10_std": None,
                "ndcg10_stderr": None,
                "ndcg10_ci90_low": None,
                "ndcg10_ci90_high": None,
                "ndcg10_ci95_low": None,
                "ndcg10_ci95_high": None,
                "mrr10": None,
                "r10": None,
            }

            for metric, mean_val in metrics.items():
                export_col = METRIC_EXPORT_MAP.get(metric.lower())
                if not export_col:
                    continue
                row[export_col] = round(float(mean_val) * METRIC_SCALE, 4)

            if not any(
                row[col] is not None
                for col in (
                    "ndcg10",
                    "mrr10",
                    "r10",
                    "ndcg10_std",
                    "ndcg10_stderr",
                    "ndcg10_ci90_low",
                    "ndcg10_ci90_high",
                    "ndcg10_ci95_low",
                    "ndcg10_ci95_high",
                )
            ):
                continue
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    out = pd.DataFrame(rows)
    out["perquery_ndcg10"] = out["result_id"].map(perquery_map)
    out["mix_ratio_sort"] = pd.to_numeric(out["mix_ratio"], errors="coerce")
    out = out.sort_values(
        ["pair","doc_mix","method","mix_ratio_sort","mix_ratio","model","source_file"],
        kind="mergesort"
    ).reset_index(drop=True)
    out = out.drop(columns=["mix_ratio_sort"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root_dir",
        nargs="?",
        default=str(DEFAULT_RESULT_ROOT),
        help=f"Path to the results directory (default: {DEFAULT_RESULT_ROOT})"
    )
    ap.add_argument(
        "-o","--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})"
    )
    ap.add_argument(
        "--processed-out",
        type=str,
        default=str(DEFAULT_PROCESSED_PATH),
        help=f"Output path for processed summary (default: {DEFAULT_PROCESSED_PATH})"
    )
    ap.add_argument(
        "--no-processed",
        action="store_true",
        help="Skip writing the processed summary CSV."
    )
    args = ap.parse_args()

    root = Path(args.root_dir)
    df = collect_results(root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = df[EXPORT_COLUMNS] if all(c in df.columns for c in EXPORT_COLUMNS) else df
    export_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (rows={len(export_df)})")

    if not args.no_processed:
        processed = compute_full_summary(load_summary_dataframe(df))
        processed_out = Path(args.processed_out)
        processed_out.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(processed_out, index=False)
        print(f"Wrote: {processed_out} (rows={len(processed)})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EN-ZH Embedding Space Analysis (EN / ZH / CM with bands)
========================================================

Given:
  - English queries (TSV: qid<TAB>text)
  - Chinese queries (TSV: qid<TAB>text)
  - 3 code-mixed bands (TSVs: qid<TAB>text)
  - Optional qids-common TSV listing qids present across all 3 bands

This script:
  1) Aligns triplets per qid&band (respecting qids-common).
  2) Embeds texts (default: BAAI/bge-m3), optional ABTT.
  3) Computes geometry metrics:
       r (on-axis ZH→EN), δ (off-axis), α/residual (linear mix fit).
  4) Trains EN-vs-ZH language probe; projects π for EN/CM/ZH.
  5) Hubness + anisotropy diagnostics.
  6) Visualisations:
       viz_umap_interactive.html / viz_tsne_interactive.html (interactive 3D; CM colored by band, EN/ZH shown once)
  7) Writes report.md and CSVs.

sample usage:
python en_zh_embedding_space_analysis.py \
  --en_file data/mmarco_dev/queries.en.tsv \
  --zh_file data/mmarco_dev/queries.zh.tsv \
  --cm_files data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm0-20.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm20-40.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm40-60.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm60-80.tsv data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm80-100.tsv \
  --cm_labels 0-20 20-40 40-60 60-80 80-100 \
  --qids_common_file data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv \
  --model_name BAAI/bge-m3 \
  --output_dir artifacts/analysis/en_zh_embedding_space\
  --neighbors_k 10 --sample_plot_n 100 --seed 42 \
  --clean_drop_qid_any_outlier
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Dimensionality reduction
try:
    import umap  # type: ignore
except Exception:
    umap = None
from sklearn.manifold import TSNE

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("ERROR: sentence-transformers is required. pip install sentence-transformers", file=sys.stderr)
    raise

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Utilities ---

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")

def set_logging(verbosity: int = 0):
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

def parse_tsv(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip("\n\r")
            if not line:
                continue
            if "\t" not in line:
                logging.warning("No TAB on line %d in %s; skipping.", i, path)
                continue
            qid, text = line.split("\t", 1)
            qid = qid.strip()
            text = text.strip()
            if not qid or not text:
                logging.warning("Empty qid/text on line %d in %s; skipping.", i, path)
                continue
            data[qid] = text
    logging.info("Loaded %d rows from %s", len(data), path)
    return data

def load_common_qids(path: Path) -> Set[str]:
    qids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip("\n\r")
            if not line:
                continue
            qid = line.split("\t", 1)[0].split(",", 1)[0].strip()
            if qid:
                qids.add(qid)
    logging.info("Loaded %d qids from %s", len(qids), path)
    return qids

def autodetect_qids_common(cm_paths: List[Path]) -> Optional[Path]:
    if not cm_paths:
        return None
    base_dir = cm_paths[0].parent
    for name in ("qids-common.tsv", "qids_common.tsv"):
        cand = base_dir / name
        if cand.exists():
            logging.info("Auto-detected common qids file: %s", cand)
            return cand
    logging.info("No qids-common file auto-detected in %s", base_dir)
    return None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def abtt_remove_top_pcs(X: np.ndarray, n_remove: int) -> Tuple[np.ndarray, np.ndarray]:
    """All-but-the-top (Mu & Viswanath, 2018)."""
    if n_remove <= 0:
        return X, np.zeros((0, X.shape[1]))
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_remove, svd_solver="auto", random_state=0)
    pca.fit(Xc)
    U = pca.components_
    proj = Xc @ U.T @ U
    Xab = Xc - proj
    return Xab, U

def embed_texts(model, texts: List[str], batch_size: int = 256, normalize: bool = True) -> np.ndarray:
    logging.info("Embedding %d texts with model=%s", len(texts), getattr(model, "model_card", "SentenceTransformer"))
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return vecs.astype(np.float32)

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T

def spearmanr_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    s = pd.Series(a).rank()
    t = pd.Series(b).rank()
    rho = s.corr(t, method="pearson")
    return float(rho), float("nan")

def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-9
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

# --- CLI ---

@dataclass
class Args:
    en_file: Path
    zh_file: Path
    cm_files: List[Path]
    cm_labels: List[str]
    model_name: str
    output_dir: Path
    abtt: int
    neighbors_k: int
    sample_plot_n: int
    seed: int
    verbose: int
    max_neighbors_vectors: int
    qids_common_file: Optional[Path]
    # New: outlier detection controls
    outlier_delta_mad: float
    outlier_cos_percentile: float
    outlier_r_margin: float
    clean_drop_qid_any_outlier: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Embedding-space analyses for code-mixed queries (EN/ZH/CM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--en_file", type=Path, required=True, help="TSV with EN queries: id<TAB>text")
    p.add_argument("--zh_file", type=Path, required=True, help="TSV with ZH queries: id<TAB>text")
    p.add_argument("--cm_files",  type=Path, nargs="+", required=True, help="TSVs for CM bands")
    p.add_argument("--cm_labels", type=str,  nargs="+",                help="Labels for CM bands (same length as --cm_files)")
    p.add_argument("--model_name", type=str, default="BAAI/bge-m3", help="Sentence-Transformer model")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "analysis" / "en_zh_embedding_space_out",
    )
    p.add_argument("--abtt", type=int, default=0, help="Remove top-N principal components (ABTT)")
    p.add_argument("--neighbors_k", type=int, default=10, help="k for neighbor diagnostics")
    p.add_argument("--sample_plot_n", type=int, default=1200, help="Max qids to include in interactive visualisations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--max_neighbors_vectors", type=int, default=20000, help="Cap vectors for neighbor diagnostics")
    p.add_argument("--qids_common_file", type=Path, default=None, help="Optional TSV with line-separated qids to restrict to; auto-detects 'qids-common.tsv' in the first CM folder if present.")

    # New: outlier detection controls
    p.add_argument('--outlier_delta_mad', type=float, default=3, help='Robust z-threshold (MAD-based) on normalized δ')
    p.add_argument('--outlier_cos_percentile', type=float, default=5.0, help='Percentile threshold on min cosine (cm↔en/zh)')
    p.add_argument('--outlier_r_margin', type=float, default=0.25, help='Allowable margin outside [0,1] for r before flagging')
    p.add_argument("--clean_drop_qid_any_outlier", action="store_true",
               help="After outlier detection, drop any qid that has at least one outlier row (strict clean).")

    a = p.parse_args()

    if a.cm_labels is None:
        # auto-label from file stems if user didn’t pass labels
        auto_labels = [Path(p).stem for p in a.cm_files]
    else:
        auto_labels = list(a.cm_labels)

    if len(a.cm_files) != len(auto_labels):
        p.error(f"--cm_files ({len(a.cm_files)}) and --cm_labels ({len(auto_labels)}) must have the same length.")


    return Args(
        en_file=a.en_file, zh_file=a.zh_file, cm_files=list(a.cm_files), cm_labels=auto_labels,
        model_name=a.model_name, output_dir=a.output_dir, abtt=a.abtt,
        neighbors_k=a.neighbors_k, sample_plot_n=a.sample_plot_n, seed=a.seed,
        verbose=a.verbose, max_neighbors_vectors=a.max_neighbors_vectors,
        qids_common_file=a.qids_common_file,
        outlier_delta_mad=a.outlier_delta_mad, outlier_cos_percentile=a.outlier_cos_percentile, outlier_r_margin=a.outlier_r_margin, clean_drop_qid_any_outlier=a.clean_drop_qid_any_outlier
    )
    
def load_and_align(en_path: Path, zh_path: Path, cm_paths: List[Path], cm_labels: List[str],
                   qids_common: Optional[Set[str]]) -> pd.DataFrame:
    """Load EN, ZH, and three CM band files; align by qid; optionally filter to qids_common; keep only qids present in all bands."""
    en = parse_tsv(en_path)
    zh = parse_tsv(zh_path)

    rows = []
    for path, label in zip(cm_paths, cm_labels):
        cm = parse_tsv(path)
        base_common = set(cm.keys()) & set(en.keys()) & set(zh.keys())
        if qids_common is not None:
            common = base_common & qids_common
            dropped_by_common = len(base_common - common)
            logging.info("[%s] Restricting to qids-common: kept %d, dropped %d vs base_common=%d",
                         label, len(common), dropped_by_common, len(base_common))
        else:
            common = base_common
            logging.info("[%s] Using base common (EN∩ZH∩CM): %d", label, len(common))

        dropped_missing = len(cm) - len(base_common)
        if dropped_missing > 0:
            logging.warning("[%s] Dropping %d CM rows without both EN and ZH endpoints.", label, dropped_missing)

        for qid in common:
            rows.append({"qid": qid, "band": label, "en": en[qid], "zh": zh[qid], "cm": cm[qid]})

    df = pd.DataFrame(rows)
    logging.info("Aligned triplets (pre-check): %d rows", len(df))

    # Ensure each qid appears in all N bands
    expected_bands = set(cm_labels)
    counts = df.groupby("qid")["band"].nunique()
    full_qids = counts[counts == len(expected_bands)].index
    if len(full_qids) < counts.size:
        logging.warning("Dropping %d qids that are not present in all bands.", counts.size - len(full_qids))
        df = df[df["qid"].isin(full_qids)].reset_index(drop=True)

    logging.info("Aligned triplets (final): %d rows across %d bands", len(df), len(cm_labels))
    return df

def count_chars_and_tokens(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Add simple script/token stats for EN, ZH, CM columns using the model's tokenizer."""
    def char_counts(s: str):
        n_han = len(HAN_RE.findall(s))
        n_lat = len(LATIN_RE.findall(s))
        return n_han, n_lat, len(s)

    stats = []
    for col in ["en", "zh", "cm"]:
        n_han, n_lat, n_chars, n_toks = [], [], [], []
        for txt in tqdm(df[col].tolist(), desc=f"Tokenizing {col}"):
            han, lat, total_chars = char_counts(txt)
            toks = tokenizer.tokenize(txt)
            n_han.append(han); n_lat.append(lat); n_chars.append(total_chars); n_toks.append(len(toks))
        stats.append(pd.DataFrame({
            f"{col}_han": n_han, f"{col}_latin": n_lat, f"{col}_chars": n_chars, f"{col}_tokens": n_toks
        }))
    return pd.concat([df.reset_index(drop=True)] + stats, axis=1)

# --- Core math ---

def compute_line_metrics(e_en: np.ndarray, e_zh: np.ndarray, e_cm: np.ndarray) -> Tuple[float, float, float, float]:
    """
    diff = ZH - EN; d = ||diff||
    u = diff / d
    p = <CM - EN, u>
    r = p / d                  # on-axis position; r in [0,1] means CM between endpoints
    δ = ||(CM - EN) - p*u||    # perpendicular distance to the axis
    """
    diff = e_zh - e_en
    d = np.linalg.norm(diff)
    if d < 1e-9:
        return float("nan"), float("nan"), float("nan"), float("nan")
    u = diff / d
    p = float(np.dot(e_cm - e_en, u))
    r = p / d
    delta = float(np.linalg.norm((e_cm - e_en) - p * u))
    return r, delta, p, d

def linear_reconstruction(e_en: np.ndarray, e_zh: np.ndarray, e_cm: np.ndarray) -> Tuple[float, float, float]:
    """Fit α in e_cm ≈ α·e_en + (1-α)·e_zh; return α, residual norm, local R²."""
    a = e_en - e_zh
    b = e_cm - e_zh
    denom = float(np.dot(a, a))
    if denom < 1e-12:
        return float("nan"), float("nan"), float("nan")
    alpha = float(np.dot(a, b) / denom)
    resid_vec = b - alpha * a
    resid = float(np.linalg.norm(resid_vec))
    b_norm = float(np.linalg.norm(b))
    r2 = 1.0 - (resid ** 2) / (b_norm ** 2 + 1e-12)
    return alpha, resid, r2

def train_language_probe(E_en: np.ndarray, E_zh: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, float]:
    """Train linear EN-vs-ZH classifier; return unit normal vector and accuracy."""
    X = np.vstack([E_en, E_zh])
    y = np.array([1] * len(E_en) + [0] * len(E_zh))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", n_jobs=1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    w = clf.coef_.ravel()
    w_norm = w / (np.linalg.norm(w) + 1e-12)
    return w_norm, acc

def neighbor_diagnostics(emb: np.ndarray, ids: List[str], k: int, max_vectors: int, seed: int) -> Dict[str, float]:
    """Hubness-like stats from top-k cosine neighbors (in-degree distribution)."""
    n_all = emb.shape[0]
    if n_all > max_vectors:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_all, size=max_vectors, replace=False)
        E = emb[idx]
        ids = [ids[i] for i in idx]
        logging.warning("Subsampled neighbor diagnostics to %d of %d vectors.", max_vectors, n_all)
    else:
        E = emb

    S = cosine_sim(E, E)
    np.fill_diagonal(S, -1.0)
    k_eff = min(k, S.shape[1] - 1)
    topk_idx = np.argpartition(-S, kth=k_eff, axis=1)[:, :k_eff]
    row_indices = np.arange(S.shape[0])[:, None]
    topk_sorted = np.argsort(-S[row_indices, topk_idx], axis=1)
    topk_idx = topk_idx[row_indices, topk_sorted]

    indeg = np.zeros(S.shape[0], dtype=np.int32)
    for nbrs in topk_idx:
        for j in nbrs:
            indeg[j] += 1

    gini = gini_coefficient(indeg.astype(float))
    indeg_sorted = np.sort(indeg)[::-1]
    top1pct = max(1, int(0.01 * len(indeg_sorted)))
    frac_top1pct = float(indeg_sorted[:top1pct].sum() / max(1, indeg_sorted.sum()))
    return {
        "n_vectors_used": int(E.shape[0]),
        "gini_indegree": gini,
        "frac_mass_top1pct": frac_top1pct,
        "avg_indegree": float(indeg.mean()),
        "max_indegree": float(indeg.max())
    }

# --- Visualizations ---


def _prepare_plot_records(df_plot: pd.DataFrame, E_map: Dict[Tuple[str, str, str], np.ndarray],
                          cm_labels: List[str]) -> Tuple[List[Dict[str, object]], List[List[int]]]:
    records: List[Dict[str, object]] = []
    connectors: List[List[int]] = []

    for qid, group in df_plot.groupby("qid"):
        try:
            vec_en = E_map[(qid, "", "en")]
            vec_zh = E_map[(qid, "", "zh")]
        except KeyError:
            logging.debug("Skipping qid %s for interactive plots (missing EN/ZH)", qid)
            continue

        idx_sequence: List[int] = []

        idx_sequence.append(len(records))
        records.append({
            "qid": qid,
            "role": "EN",
            "band": "EN",
            "vector": vec_en,
        })

        available_bands = set(group["band"].astype(str))
        for band in cm_labels:
            if band not in available_bands:
                continue
            vec_cm = E_map.get((qid, band, "cm"))
            if vec_cm is None:
                continue
            idx_sequence.append(len(records))
            records.append({
                "qid": qid,
                "role": "CM",
                "band": band,
                "vector": vec_cm,
            })

        idx_sequence.append(len(records))
        records.append({
            "qid": qid,
            "role": "ZH",
            "band": "ZH",
            "vector": vec_zh,
        })

        if len(idx_sequence) >= 2:
            connectors.append(idx_sequence)

    return records, connectors


def _write_interactive_plot(coords: np.ndarray, records: List[Dict[str, object]],
                            connectors: List[List[int]], out_html: Path, title: str,
                            cm_labels: List[str]) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as e:
        logging.warning("Plotly unavailable; skipping interactive plot %s (%s)", out_html, e)
        return

    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")

    df_points = pd.DataFrame({
        "idx": np.arange(len(records), dtype=int),
        "qid": [str(rec["qid"]) for rec in records],
        "role": [rec["role"] for rec in records],
        "band": [rec["band"] for rec in records],
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })

    color_order = ["EN"] + [b for b in cm_labels if (df_points["band"] == b).any()] + ["ZH"]
    color_palette = (px.colors.qualitative.Plotly + px.colors.qualitative.Set2 +
                     px.colors.qualitative.Dark24)
    color_map: Dict[str, str] = {}
    for idx, band in enumerate(color_order):
        if band == "EN":
            color_map[band] = "#555555"
        elif band == "ZH":
            color_map[band] = "#999999"
        else:
            color_map[band] = color_palette[idx % len(color_palette)]

    fig = px.scatter_3d(
        df_points,
        x="x",
        y="y",
        z="z",
        color="band",
        symbol="role",
        hover_data={"qid": True, "role": True, "band": True},
        category_orders={"band": color_order, "role": ["EN", "CM", "ZH"]},
        color_discrete_map=color_map,
        title=title,
    )

    df_points = df_points.set_index("idx")

    for seq in connectors:
        if len(seq) < 2:
            continue
        try:
            subset = df_points.loc[seq]
        except KeyError:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.35)", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
    fig.update_layout(scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    logging.info("Saved interactive plot: %s", out_html)


def make_umap_tsne_plots(df: pd.DataFrame, E_map: Dict[Tuple[str, str, str], np.ndarray],
                         out_dir: Path, cm_labels: List[str], sample_qids: int = 1200,
                         seed: int = 42):
    if df.empty:
        logging.warning("No data for visualization; skipping plots.")
        return

    unique_qids = df["qid"].unique().tolist()
    if len(unique_qids) > sample_qids:
        rng = np.random.default_rng(seed)
        chosen_qids = set(rng.choice(unique_qids, size=sample_qids, replace=False))
        df_plot = df[df["qid"].isin(chosen_qids)].copy()
        logging.info("Sampled %d / %d qids for interactive plots.", len(chosen_qids), len(unique_qids))
    else:
        df_plot = df.copy()

    records, connectors = _prepare_plot_records(df_plot, E_map, cm_labels)
    if not records:
        logging.warning("No records available for visualization; skipping interactive plots.")
        return

    X = np.vstack([rec["vector"] for rec in records])

    if umap is not None:
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
            n_components=3,
        )
        coords_umap = reducer.fit_transform(X)
        _write_interactive_plot(
            coords_umap,
            records,
            connectors,
            out_dir / "viz_umap_interactive.html",
            "UMAP (cosine) — interactive",
            cm_labels,
        )
    else:
        logging.warning("umap-learn not installed; skipping UMAP interactive plot.")

    try:
        tsne = TSNE(
            n_components=3,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            metric="cosine",
        )
    except TypeError:
        tsne = TSNE(
            n_components=3,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        )
    coords_tsne = tsne.fit_transform(X)

    _write_interactive_plot(
        coords_tsne,
        records,
        connectors,
        out_dir / "viz_tsne_interactive.html",
        "t-SNE — interactive",
        cm_labels,
    )


# --- Main ---

def main():
    args = parse_args()
    set_logging(args.verbose)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", args.output_dir)

    # qids-common
    qids_common_set: Optional[Set[str]] = None
    qids_common_path: Optional[Path] = None
    if args.qids_common_file is not None:
        qids_common_path = args.qids_common_file
        if qids_common_path.exists():
            qids_common_set = load_common_qids(qids_common_path)
        else:
            logging.error("Provided --qids_common_file does not exist: %s", qids_common_path)
            sys.exit(1)
    else:
        auto = autodetect_qids_common(args.cm_files)
        if auto is not None:
            qids_common_path = auto
            qids_common_set = load_common_qids(auto)

    # Load & align
    df = load_and_align(args.en_file, args.zh_file, args.cm_files, args.cm_labels, qids_common_set)
    if df.empty:
        logging.error("No aligned data after applying filters.")
        sys.exit(1)

    # Model & tokenizer
    logging.info("Loading embedding model: %s", args.model_name)
    import torch
    model = SentenceTransformer(
        args.model_name,
        device="cuda:1" if torch.cuda.is_available() else "cpu"
    )
    try:
        tokenizer = model.tokenizer
    except Exception:
        tokenizer = None
        logging.warning("Tokenizer not found on model; token counts will be skipped.")
    if tokenizer is not None:
        df = count_chars_and_tokens(df, tokenizer)
    else:
        for col in ["en", "zh", "cm"]:
            df[f"{col}_han"] = 0
            df[f"{col}_latin"] = 0
            df[f"{col}_chars"] = df[col].apply(len)
            df[f"{col}_tokens"] = 0

    # Embed unique (qid,band,kind); EN/ZH deduped
    key_map: Dict[Tuple[str, str, str], str] = {}
    for _, row in df.iterrows():
        qid = row["qid"]; band = row["band"]
        key_map[(qid, "", "en")] = row["en"]
        key_map[(qid, "", "zh")] = row["zh"]
        key_map[(qid, band, "cm")] = row["cm"]
    key_meta: List[Tuple[str, str, str]] = list(key_map.keys())
    key_texts: List[str] = [key_map[k] for k in key_meta]
    logging.info("Total unique (qid,band,kind) to embed: %d (EN/ZH deduped, CM band-specific)", len(key_texts))

    E = embed_texts(model, key_texts, batch_size=128, normalize=True)
    d = E.shape[1]

    # ABTT (optional)
    if args.abtt > 0:
        logging.info("Applying ABTT: removing top-%d principal components.", args.abtt)
        E, _ = abtt_remove_top_pcs(E, args.abtt)
        E = l2_normalize(E)

    # Map back
    E_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    for key, vec in zip(key_meta, E):
        E_map[key] = vec

    # Per-row metrics
    results: List[Dict[str, object]] = []
    r_by = {lab: [] for lab in args.cm_labels}
    cos_en_by = {lab: [] for lab in args.cm_labels}
    cos_zh_by = {lab: [] for lab in args.cm_labels}
    delta_by = {lab: [] for lab in args.cm_labels}
    alpha_by = {lab: [] for lab in args.cm_labels}
    resid_by = {lab: [] for lab in args.cm_labels}
    r2_by = {lab: [] for lab in args.cm_labels}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing metrics"):
        qid = row["qid"]; band = row["band"]
        e_en = E_map[(qid, "", "en")]
        e_zh = E_map[(qid, "", "zh")]
        e_cm = E_map[(qid, band, "cm")]

        r, delta, p, d_axis = compute_line_metrics(e_en, e_zh, e_cm)
        delta_rel = float(delta / (d_axis + 1e-12)) if np.isfinite(d_axis) else float("nan")
        alpha, resid, r2 = linear_reconstruction(e_en, e_zh, e_cm)
        # Cosine similarities (assumes embeddings are L2-normalized after ABTT)
        cos_en = float(np.dot(e_cm, e_en))
        cos_zh = float(np.dot(e_cm, e_zh))

        r_by[band].append(r)
        delta_by[band].append(delta)
        alpha_by[band].append(alpha)
        resid_by[band].append(resid)
        r2_by[band].append(r2)
        cos_en_by[band].append(cos_en)
        cos_zh_by[band].append(cos_zh)

        results.append({
            "qid": qid, "band": band,
            "r_between_0_1": (0.0 <= r <= 1.0) if not math.isnan(r) else False,
            "r": r, "delta": delta, "delta_rel": delta_rel, "p": p, "d_axis": d_axis,
            "alpha": alpha, "residual_norm": resid, "r2_local": r2,
            "cos_en": cos_en, "cos_zh": cos_zh,
            "en_tokens": row.get("en_tokens", 0), "zh_tokens": row.get("zh_tokens", 0), "cm_tokens": row.get("cm_tokens", 0),
            "en_han": row.get("en_han", 0), "en_latin": row.get("en_latin", 0),
            "zh_han": row.get("zh_han", 0), "zh_latin": row.get("zh_latin", 0),
            "cm_han": row.get("cm_han", 0), "cm_latin": row.get("cm_latin", 0),
        })

    res_df = pd.DataFrame(results)
    out_csv = Path(args.output_dir) / "per_query_metrics.csv"
    if "delta_rel" not in res_df.columns:
        res_df["delta_rel"] = res_df["delta"] / (res_df["d_axis"] + 1e-12)
    res_df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved per-query metrics: %s", out_csv)
    # Cosine similarity summary per qid (wide format: cos_en/cos_zh by band)
    try:
        piv = res_df.pivot_table(index='qid', columns='band', values=['cos_en','cos_zh'], aggfunc='mean')
        piv.columns = [f"{a}_{b}" for a,b in piv.columns]
        piv = piv.reset_index()
        piv.to_csv(Path(args.output_dir) / 'cosine_by_qid.csv', index=False, encoding='utf-8')
        logging.info('Saved cosine_by_qid.csv')
    except Exception as e:
        logging.warning('Could not create cosine_by_qid.csv: %s', e)

    # Outliers (robust detection)
    res_df["delta_rel"] = res_df.get("delta_rel", res_df["delta"] / (res_df["d_axis"] + 1e-12))
    # (Optional) keep old name as alias if referenced elsewhere
    res_df["delta_over_axis"] = res_df["delta_rel"]

    nd  = res_df["delta_rel"].replace([np.inf, -np.inf], np.nan)
    med = float(np.nanmedian(nd))
    mad = float(np.nanmedian(np.abs(nd - med)) + 1e-12)
    res_df["z_delta"] = 0.6745 * (res_df["delta_rel"] - med) / mad

    res_df["min_cos"] = np.minimum(res_df["cos_en"], res_df["cos_zh"])
    cos_thresh = float(np.nanpercentile(res_df["min_cos"], args.outlier_cos_percentile))
    # r outside [0,1] by margin
    too_far_r = (res_df["r"] < (0.0 - args.outlier_r_margin)) | (res_df["r"] > (1.0 + args.outlier_r_margin))
    far_delta = res_df["z_delta"] > args.outlier_delta_mad
    low_cos = res_df["min_cos"] < cos_thresh
    tiny_axis = res_df["d_axis"] < 1e-3  # degenerate EN–ZH pair

    res_df["is_outlier"] = (too_far_r | far_delta | low_cos | tiny_axis)

    # Reasons
    reasons = []
    for tf, fd, lc, ta in zip(too_far_r, far_delta, low_cos, tiny_axis):
        r = []
        if tf: r.append("r_outside")
        if fd: r.append("delta_mad")
        if lc: r.append("low_cos")
        if ta: r.append("tiny_axis")
        reasons.append(",".join(r) if r else "")
    res_df["outlier_reason"] = reasons

    outliers = res_df[res_df["is_outlier"]].copy()
    outliers_path = Path(args.output_dir) / "outliers.csv"
    outliers.to_csv(outliers_path, index=False, encoding="utf-8")
    # Unique qids
    outlier_qids = sorted(set(outliers["qid"].astype(str).tolist()))
    (Path(args.output_dir) / "outliers_qids.txt").write_text("\n".join(outlier_qids), encoding="utf-8")
    logging.info("Outliers saved (%d rows, %d unique qids). Cosine p%%=%.1f -> %.4f; MAD Δ-thresh=%.2f",
                 len(outliers), len(outlier_qids), args.outlier_cos_percentile, cos_thresh, args.outlier_delta_mad)

    clean_rowwise = res_df[~res_df["is_outlier"]].copy()

    # QID-wise strict clean (makes n equal across bands AFTER cleaning):
    if args.clean_drop_qid_any_outlier:
        bad_qids = set(res_df.loc[res_df["is_outlier"], "qid"].astype(str))
        clean_qidwise = res_df[~res_df["qid"].astype(str).isin(bad_qids)].copy()
    else:
        clean_qidwise = clean_rowwise  # default to row-wise if flag not set

    # (Optional) Save for downstream inspection
    (Path(args.output_dir) / "per_query_metrics_clean_rowwise.csv").write_text(
        clean_rowwise.to_csv(index=False), encoding="utf-8"
    )
    (Path(args.output_dir) / "per_query_metrics_clean_qidwise.csv").write_text(
        clean_qidwise.to_csv(index=False), encoding="utf-8"
    )

    # Language probe: use each qid once
    qids_list = df["qid"].tolist()
    seen: Set[str] = set()
    uq: List[str] = []
    for q in qids_list:
        if q not in seen:
            uq.append(q)
            seen.add(q)
    E_en = np.stack([E_map[(qid, "", "en")] for qid in uq])
    E_zh = np.stack([E_map[(qid, "", "zh")] for qid in uq])
    v_hat, probe_acc = train_language_probe(E_en, E_zh, seed=args.seed)
    logging.info("Language probe accuracy (held-out): %.4f", probe_acc)

    # Projections
    def lang_proj(e): return float(np.dot(e, v_hat))
    proj_rows = []
    for _, row in df.iterrows():
        qid = row["qid"]; band = row["band"]
        proj_rows.append({
            "qid": qid, "band": band,
            "pi_en": lang_proj(E_map[(qid, "", "en")]),
            "pi_cm": lang_proj(E_map[(qid, band, "cm")]),
            "pi_zh": lang_proj(E_map[(qid, "", "zh")]),
        })
    proj_df = pd.DataFrame(proj_rows)
    out_csv2 = Path(args.output_dir) / "language_direction_projections.csv"
    proj_df.to_csv(out_csv2, index=False, encoding="utf-8")
    logging.info("Saved language-direction projections: %s", out_csv2)

    # Neighbor / hubness: EN/ZH once per qid; CM per band
    ids_all: List[str] = []
    E_all_list: List[np.ndarray] = []
    seen_mono: Set[Tuple[str, str]] = set()
    for _, row in df.iterrows():
        qid = row["qid"]; band = row["band"]
        if (qid, "en") not in seen_mono:
            ids_all.append(f"{qid}||en")
            E_all_list.append(E_map[(qid, "", "en")])
            seen_mono.add((qid, "en"))
        if (qid, "zh") not in seen_mono:
            ids_all.append(f"{qid}||zh")
            E_all_list.append(E_map[(qid, "", "zh")])
            seen_mono.add((qid, "zh"))
        ids_all.append(f"{qid}|{band}|cm")
        E_all_list.append(E_map[(qid, band, "cm")])
    E_all = np.vstack(E_all_list)
    hub_stats = neighbor_diagnostics(E_all, ids_all, k=args.neighbors_k, max_vectors=args.max_neighbors_vectors, seed=args.seed)
    out_json = Path(args.output_dir) / "hubness_stats.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(hub_stats, f, indent=2, ensure_ascii=False)
    logging.info("Saved hubness stats: %s", out_json)

    # Anisotropy
    rng = np.random.default_rng(args.seed)
    n_pairs = min(50000, E_all.shape[0] * 10)
    idx1 = rng.integers(0, E_all.shape[0], size=n_pairs)
    idx2 = rng.integers(0, E_all.shape[0], size=n_pairs)
    avg_cos = float(np.mean(np.sum(E_all[idx1] * E_all[idx2], axis=1)))
    ani_json = Path(args.output_dir) / "anisotropy.json"
    with ani_json.open("w", encoding="utf-8") as f:
        json.dump({"avg_random_cosine": avg_cos}, f, indent=2)
    logging.info("Saved anisotropy estimate: %s (avg_random_cosine=%.6f)", ani_json, avg_cos)

    # Report
    lines: List[str] = []
    lines.append("# Code-Mixed Embedding Space Analysis Report\n")
    lines.append(f"- Model: `{args.model_name}`")
    lines.append(f"- ABTT removed PCs: `{args.abtt}`")
    lines.append(f"- Total aligned rows: `{len(df)}` across bands: `{', '.join(args.cm_labels)}`")
    if qids_common_path is not None and qids_common_set is not None:
        lines.append(f"- qids-common file: `{qids_common_path}` (|qids|={len(qids_common_set)})")
    lines.append("")

    mix_midpoint: Dict[str, float] = {}
    for lab in args.cm_labels:
        try:
            lo, hi = lab.split("-")
            mid = (float(lo) + float(hi)) / 200.0  # normalize to 0..1
        except Exception:
            mid = float("nan")
        mix_midpoint[lab] = mid

    lines.append("## Per-band summaries")

    # ---- Config for outlier detector
    if "is_outlier" in res_df.columns:
        cos_thresh = float(np.nanpercentile(res_df["min_cos"], args.outlier_cos_percentile))
        lines.append("## Outlier detector configuration")
        lines.append(f"- `z_delta` MAD threshold: `{args.outlier_delta_mad}`")
        lines.append(f"- `min_cos` percentile: `{args.outlier_cos_percentile}%` → threshold = `{cos_thresh:.4f}`")
        lines.append(f"- `r` margin outside [0,1]: `{args.outlier_r_margin}`")
        lines.append("")

    # ---- Counts BEFORE removal
    lines.append("## BEFORE outlier removal (all rows)")
    pre_rows_by_band = res_df.groupby("band").size().to_dict()
    pre_qids_by_band = res_df.groupby("band")["qid"].nunique().to_dict()
    lines.append("- **Row counts by band (rows = qid×band):**")
    for b in args.cm_labels:
        lines.append(f"  - `{b}`: rows=`{pre_rows_by_band.get(b, 0)}`, unique qids=`{pre_qids_by_band.get(b, 0)}`")
    lines.append("")

    # ---- Per-band metrics BEFORE
    lines.append("### Per-band summaries (BEFORE)")
    for lab in args.cm_labels:
        sub = res_df[res_df["band"] == lab]
        r_arr  = sub["r"].to_numpy(dtype=float)
        d_arr  = sub["delta"].to_numpy(dtype=float)
        a_arr  = sub["alpha"].to_numpy(dtype=float)
        res_arr= sub["residual_norm"].to_numpy(dtype=float)
        r2_arr = sub["r2_local"].to_numpy(dtype=float)
        n_b = len(r_arr)
        frac_between = float(np.mean((r_arr >= 0.0) & (r_arr <= 1.0))) if n_b else float("nan")
        def tmean_local(x):
            x = x[~np.isnan(x)]
            if x.size == 0: return float("nan")
            lo, hi = np.quantile(x, [0.05, 0.95]); xx = x[(x >= lo) & (x <= hi)]
            return float(np.mean(xx)) if xx.size else float("nan")
        lines.append(f"#### Band `{lab}`  (n={n_b})")
        lines.append(f"- mean r: `{np.nanmean(r_arr):.4f}` | trimmed mean r: `{tmean_local(r_arr):.4f}` | median r: `{np.nanmedian(r_arr):.4f}` | frac r∈[0,1]: `{frac_between:.3f}`")
        lines.append(f"- mean δ: `{np.nanmean(d_arr):.4f}` | trimmed mean δ: `{tmean_local(d_arr):.4f}` | median δ: `{np.nanmedian(d_arr):.4f}`")
        dr = sub["delta_rel"].to_numpy(dtype=float)
        lines.append(f"- mean δ_rel (δ/|EN–ZH|): `{np.nanmean(dr):.4f}` | median δ_rel: `{np.nanmedian(dr):.4f}`")
        lines.append(f"- mean α: `{np.nanmean(a_arr):.4f}` | mean residual: `{np.nanmean(res_arr):.4f}` | mean local R²: `{np.nanmean(r2_arr):.4f}`")
    lines.append("")

    # ---- Cosine BEFORE
    lines.append("### Cosine similarity (CM vs EN/ZH) by band (BEFORE)")
    for lab in args.cm_labels:
        cen = res_df.loc[res_df["band"] == lab, "cos_en"].to_numpy(dtype=float)
        czh = res_df.loc[res_df["band"] == lab, "cos_zh"].to_numpy(dtype=float)
        def qstats(x):
            x = x[~np.isnan(x)]
            if x.size == 0: return ("nan","nan","nan","nan")
            return (f"{np.nanmean(x):.4f}", f"{np.nanmedian(x):.4f}",
                    f"{np.nanpercentile(x,10):.4f}", f"{np.nanpercentile(x,90):.4f}")
        m_en, md_en, p10_en, p90_en = qstats(cen); m_zh, md_zh, p10_zh, p90_zh = qstats(czh)
        lines.append(f"- **{lab}**: cos(cm,en) mean/median p10–p90: `{m_en}` / `{md_en}` [{p10_en}–{p90_en}] | cos(cm,zh): `{m_zh}` / `{md_zh}` [{p10_zh}–{p90_zh}]")
    lines.append("")

    # ---- Outlier summary
    if "is_outlier" in res_df.columns:
        n_out = int(res_df["is_outlier"].sum())
        lines.append("## Outlier detection results")
        lines.append(f"- Total outlier rows: `{n_out}` of `{len(res_df)}`; unique qids: `{res_df.loc[res_df['is_outlier'],'qid'].nunique()}`")
        by_band_rows = res_df[res_df["is_outlier"]].groupby("band").size().to_dict()
        by_band_qids = res_df[res_df["is_outlier"]].groupby("band")["qid"].nunique().to_dict()
        lines.append("- Outliers by band:")
        for b in args.cm_labels:
            lines.append(f"  - `{b}`: outlier rows=`{by_band_rows.get(b,0)}`, outlier qids=`{by_band_qids.get(b,0)}`")
        # sample qids
        sample_qids = sorted(set(res_df.loc[res_df["is_outlier"], "qid"].astype(str)))[:20]
        if sample_qids:
            lines.append(f"- Sample outlier qids (≤20): `{', '.join(sample_qids)}`")
        lines.append("")

    clean: Optional[pd.DataFrame] = None

    # ---- AFTER (row-wise clean)
    if "is_outlier" in res_df.columns:
        if args.clean_drop_qid_any_outlier:
            lines.append("## AFTER outlier removal (qid-wise strict)")
        else:
            lines.append("## AFTER outlier removal (row-wise)")
        clean = clean_qidwise if args.clean_drop_qid_any_outlier else clean_rowwise
        post_rows_by_band = clean.groupby("band").size().to_dict()
        post_qids_by_band = clean.groupby("band")["qid"].nunique().to_dict()
        lines.append("- **Row counts by band (after row-wise clean):**")
        for b in args.cm_labels:
            lines.append(f"  - `{b}`: rows=`{post_rows_by_band.get(b, 0)}`, unique qids=`{post_qids_by_band.get(b, 0)}`")
        lines.append("")

        lines.append("### Per-band summaries (AFTER, row-wise)")
        for lab in args.cm_labels:
            sub = clean[clean["band"] == lab]
            r_arr  = sub["r"].to_numpy(dtype=float)
            d_arr  = sub["delta"].to_numpy(dtype=float)
            a_arr  = sub["alpha"].to_numpy(dtype=float)
            res_arr= sub["residual_norm"].to_numpy(dtype=float)
            r2_arr = sub["r2_local"].to_numpy(dtype=float)
            n_b = len(r_arr)
            frac_between = float(np.mean((r_arr >= 0.0) & (r_arr <= 1.0))) if n_b else float("nan")
            def tmean_local(x):
                x = x[~np.isnan(x)]
                if x.size == 0: return float("nan")
                lo, hi = np.quantile(x, [0.05, 0.95]); xx = x[(x >= lo) & (x <= hi)]
                return float(np.mean(xx)) if xx.size else float("nan")
            lines.append(f"#### Band `{lab}`  (n={n_b})")
            lines.append(f"- mean r: `{np.nanmean(r_arr):.4f}` | trimmed mean r: `{tmean_local(r_arr):.4f}` | median r: `{np.nanmedian(r_arr):.4f}` | frac r∈[0,1]: `{frac_between:.3f}`")
            lines.append(f"- mean δ: `{np.nanmean(d_arr):.4f}` | trimmed mean δ: `{tmean_local(d_arr):.4f}` | median δ: `{np.nanmedian(d_arr):.4f}`")
            dr = sub["delta_rel"].to_numpy(dtype=float)
            lines.append(f"- mean δ_rel (δ/|EN–ZH|): `{np.nanmean(dr):.4f}` | median δ_rel: `{np.nanmedian(dr):.4f}`")
            lines.append(f"- mean α: `{np.nanmean(a_arr):.4f}` | mean residual: `{np.nanmean(res_arr):.4f}` | mean local R²: `{np.nanmean(r2_arr):.4f}`")
        lines.append("")

        lines.append("### Cosine similarity (CM vs EN/ZH) by band (AFTER)")
        for lab in args.cm_labels:
            cen = clean.loc[clean["band"] == lab, "cos_en"].to_numpy(dtype=float)
            czh = clean.loc[clean["band"] == lab, "cos_zh"].to_numpy(dtype=float)

            def qstats(x: np.ndarray):
                x = x[~np.isnan(x)]
                if x.size == 0:
                    return ("nan", "nan", "nan", "nan")
                return (
                    f"{np.nanmean(x):.4f}",
                    f"{np.nanmedian(x):.4f}",
                    f"{np.nanpercentile(x, 10):.4f}",
                    f"{np.nanpercentile(x, 90):.4f}",
                )

            m_en, md_en, p10_en, p90_en = qstats(cen)
            m_zh, md_zh, p10_zh, p90_zh = qstats(czh)
            lines.append(
                f"- **{lab}**: cos(cm,en) mean/median p10–p90: `{m_en}` / `{md_en}` [{p10_en}–{p90_en}] | "
                f"cos(cm,zh): `{m_zh}` / `{md_zh}` [{p10_zh}–{p90_zh}]"
            )
        lines.append("")

    summary_df = clean if clean is not None else res_df

    # Plots (post-clean if available)
    df_for_plots = df.copy()
    if clean is not None:
        if args.clean_drop_qid_any_outlier:
            keep_qids = set(clean["qid"].astype(str))
            df_for_plots = df[df["qid"].astype(str).isin(keep_qids)].copy()
        else:
            keep_pairs = set(zip(clean["qid"].astype(str), clean["band"].astype(str)))
            df_keys = list(zip(df["qid"].astype(str), df["band"].astype(str)))
            mask = [key in keep_pairs for key in df_keys]
            df_for_plots = df.loc[mask].copy()

    make_umap_tsne_plots(
        df_for_plots,
        E_map,
        Path(args.output_dir),
        args.cm_labels,
        sample_qids=args.sample_plot_n,
        seed=args.seed,
    )

    sum_rows: List[Dict[str, object]] = []
    for lab in args.cm_labels:
        sub = summary_df[summary_df["band"] == lab]
        r_arr = sub["r"].to_numpy(dtype=float)
        d_arr = sub["delta"].to_numpy(dtype=float)
        a_arr = sub["alpha"].to_numpy(dtype=float)
        res_arr = sub["residual_norm"].to_numpy(dtype=float)
        r2_arr = sub["r2_local"].to_numpy(dtype=float)
        n_b = len(r_arr)

        frac_between = float(np.mean((r_arr >= 0.0) & (r_arr <= 1.0))) if n_b else float("nan")

        def tmean(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if x.size == 0:
                return float("nan")
            lo, hi = np.quantile(x, [0.05, 0.95])
            xx = x[(x >= lo) & (x <= hi)]
            return float(np.mean(xx)) if xx.size else float("nan")

        sum_rows.append({
            "band": lab,
            "n": n_b,
            "mix_midpoint_est": mix_midpoint[lab],
            "mean_r": float(np.nanmean(r_arr)),
            "median_r": float(np.nanmedian(r_arr)),
            "trimmed_mean_r_5pct": tmean(r_arr),
            "frac_r_between_0_1": frac_between,
            "mean_delta": float(np.nanmean(d_arr)),
            "median_delta": float(np.nanmedian(d_arr)),
            "trimmed_mean_delta_5pct": tmean(d_arr),
            "mean_alpha": float(np.nanmean(a_arr)),
            "mean_residual": float(np.nanmean(res_arr)),
            "mean_local_R2": float(np.nanmean(r2_arr)),
        })

    if sum_rows:
        lines.append("## Aggregated band statistics")
        if clean is not None:
            lines.append("- Saved to `band_summaries.csv` (after outlier removal).")
        else:
            lines.append("- Saved to `band_summaries.csv` (all rows).")
        lines.append("")

    try:
        means_r = np.array([row["mean_r"] for row in sum_rows], dtype=float)
        means_d = np.array([row["mean_delta"] for row in sum_rows], dtype=float)
        mids = np.array([row["mix_midpoint_est"] for row in sum_rows], dtype=float)
        rho_r, _ = spearmanr_safe(mids, means_r)
        rho_d, _ = spearmanr_safe(mids, means_d)
        lines.append("## Correlations across bands")
        if clean is not None:
            lines.append("- Stats computed on data after outlier removal.")
        lines.append(f"- Spearman(mix_midpoint, mean r) ≈ `{rho_r:.4f}`")
        lines.append(f"- Spearman(mix_midpoint, mean δ) ≈ `{rho_d:.4f}`")
        lines.append("")
    except Exception as e:
        logging.warning("Correlation across bands failed: %s", e)

    sum_df = pd.DataFrame(sum_rows)
    sum_csv = Path(args.output_dir) / "band_summaries.csv"
    sum_df.to_csv(sum_csv, index=False, encoding="utf-8")
    logging.info("Saved band summaries: %s", sum_csv)

    lines.append("## Diagnostics & Plots")
    lines.append(f"- Hubness stats: `hubness_stats.json`")
    lines.append(f"- Anisotropy: `anisotropy.json`")
    lines.append(f"- UMAP (interactive): `viz_umap_interactive.html`")
    lines.append(f"- t-SNE (interactive): `viz_tsne_interactive.html`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- EN (gray), ZH (light gray). CM colours follow band labels.")
    lines.append("- Gray polylines connect EN → CM bands → ZH for each qid (no duplicate points).")
    lines.append("- Use r/δ/α for quantitative claims; UMAP/t-SNE are for intuition.")
    lines.append("")
    lines.append("---")
    lines.append("Generated by en_zh_embedding_space_analysis.py")

    report_md = Path(args.output_dir) / "report.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info("Saved report: %s", report_md)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        sys.exit(2)

import os
import re
import textwrap
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from matplotlib.lines import Line2D
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(PROJECT_ROOT, "artifacts", "tables", "full_mmarco_processed_results.csv")
CURVES = os.path.join(PROJECT_ROOT, "artifacts", "tables", "full_mmarco_results.csv")
ABLATION = os.path.join(PROJECT_ROOT, "artifacts", "tables", "ablation_processed_results.csv")
OUTDIR = os.path.join(PROJECT_ROOT, "assets", "figures")
RATIO_PRIMARY = os.path.join(PROJECT_ROOT, "artifacts", "tables", "unified_results.csv")
RATIO_FALLBACK = os.path.join(PROJECT_ROOT, "artifacts", "tables", "converted_results_full.csv")
EMBEDDING_BANDS = os.path.join(
    PROJECT_ROOT, "artifacts", "analysis", "en_zh_embedding_space", "band_summaries.csv"
)
EMBEDDING_BANDS_FALLBACKS = [
    os.path.join(PROJECT_ROOT, "artifacts", "analysis", "en_zh_embedding_space", "band_summaries.csv"),
    os.path.join(PROJECT_ROOT, "artifacts", "analysis", "cm_analysis_dev", "band_summaries.csv"),
]

os.makedirs(OUTDIR, exist_ok=True)

FIGSIZE = (6.5, 3.8)
HIST_COLOR = "#414C87"
EDGE_COLOR = "#666666"
LINE_ZERO_COLOR = "black"
LINE_ZERO_WIDTH = 1.6
LINE_PLOT_WIDTH = 2.5
LINE_PLOT_MARKER_SIZE = 8
JITTER_POINT_SIZE = 36
SCATTER_POINT_SIZE = 36
HEADROOM_POINT_SIZE = 36
DUMBBELL_LINE_WIDTH = 2
GRID_ALPHA = 0.2
GRID_WIDTH = 0.8
SAVEFIG_DPI = 300
PALETTE = ["#5D74A2", "#922125", "#EEC280"]
TITLE_FS = 15
LABEL_FS = 14
TICK_FS = 13
LEGEND_FS = 12
LEGEND_SMALL_FS = 12
RATIO_XTICKS = [0, 10, 30, 50, 70, 90, 100]
RATIO_FIGSIZE = (7.2, 4.8)
RATIO_TITLE_FS = 21
RATIO_LABEL_FS = 18
RATIO_TICK_FS = 16
RATIO_LEGEND_FS = 15
RATIO_LEGEND_TITLE_FS = 16
RATIO_LEGEND_TITLE = "method"
RATIO_LINE_WIDTH = 2.4
RATIO_MARKER_SIZE = 7
RATIO_HIGHLIGHT_SIZE = 12
RATIO_ANNOT_FS = 15
RATIO_ANNOT_DX_FRAC = 0.03
RATIO_ANNOT_DY_FRAC = 0.04
RATIO_ANNOT_BBOX_ALPHA = 0.75

def apply_academic_style(ax, tick_fs: int = TICK_FS):
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=GRID_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=tick_fs)

def save_fig(fig, outpath: str):
    fig.tight_layout()
    fig.savefig(outpath, dpi=SAVEFIG_DPI)
    base, ext = os.path.splitext(outpath)
    if ext.lower() == ".pdf":
        fig.savefig(f"{base}.png", dpi=SAVEFIG_DPI)
    plt.close(fig)

def load_and_filter_processed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only pairs that have all 3 doc regimes (so totals match 35*3=105).
    pair_counts = df["pair"].value_counts()
    valid_pairs = pair_counts[pair_counts == 3].index
    df = df[df["pair"].isin(valid_pairs)].copy()

    # Convenience columns
    df["en_in_index"] = df["doc_mix"].str.contains("EN")
    df["doc_lang"] = df["doc_mix"].str.replace(" docs", "", regex=False)  # works for mono; bi contains '+'
    return df

def plot_delta_distribution(df: pd.DataFrame, outpath: str):
    deltas = df["delta_ndcg"].to_numpy()

    pos = np.sum(deltas > 0)
    neg = np.sum(deltas < 0)
    zer = np.sum(deltas == 0)

    mean = float(np.mean(deltas))
    median = float(np.median(deltas))

    fig, ax = plt.subplots(figsize=RATIO_FIGSIZE)
    ax.hist(deltas, bins=25, color=HIST_COLOR, edgecolor=EDGE_COLOR, alpha=0.65, linewidth=1.1)
    ax.axvline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=3, label="0")
    ax.axvline(mean, color=PALETTE[0], linewidth=2.2, linestyle="--", zorder=3, label=f"Mean = {mean:.4f}")
    ax.axvline(median, color=PALETTE[1], linewidth=2.2, linestyle=":", zorder=3, label=f"Median = {median:.4f}")
    ax.set_xlabel(r"$\Delta$ nDCG@10 (best mid − best endpoint)", fontsize=RATIO_LABEL_FS)
    ax.set_ylabel("Number of settings", fontsize=RATIO_LABEL_FS)
    ax.set_title(f"Δ distribution (n={len(deltas)})", fontsize=RATIO_TITLE_FS)
    apply_academic_style(ax, tick_fs=RATIO_TICK_FS)
    ax.legend(frameon=False, loc="upper right", fontsize=RATIO_LEGEND_FS)
    save_fig(fig, outpath)

def plot_en_in_index_split(df: pd.DataFrame, outpath: str):
    # Two buckets
    a = df[~df["en_in_index"]]["delta_ndcg"].to_numpy()
    b = df[df["en_in_index"]]["delta_ndcg"].to_numpy()

    fig, ax = plt.subplots(figsize=RATIO_FIGSIZE)
    ax.axhline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=2)

    ratio_scale = RATIO_LABEL_FS / LABEL_FS
    point_size = JITTER_POINT_SIZE * (ratio_scale ** 2)

    # Boxplot
    box = ax.boxplot(
        [a, b],
        tick_labels=["EN absent", "EN present"],
        showfliers=False,
        patch_artist=True,
        widths=0.7,
        boxprops={"edgecolor": EDGE_COLOR, "linewidth": 1.4},
        medianprops={"color": EDGE_COLOR, "linewidth": 1.8},
        whiskerprops={"color": EDGE_COLOR, "linewidth": 1.4},
        capprops={"color": EDGE_COLOR, "linewidth": 1.4},
    )
    for patch, color in zip(box["boxes"], PALETTE[:2]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    # Jittered points
    rng = np.random.default_rng(0)
    x1 = 1 + rng.uniform(-0.35, 0.35, size=len(a))
    x2 = 2 + rng.uniform(-0.35, 0.35, size=len(b))
    ax.scatter(x1, a, s=point_size, color=PALETTE[0], alpha=0.6)
    ax.scatter(x2, b, s=point_size, color=PALETTE[1], alpha=0.6)

    ax.set_ylabel(r"$\Delta$ nDCG@10", fontsize=RATIO_LABEL_FS)
    ax.set_title("EN in index split", fontsize=RATIO_TITLE_FS)
    apply_academic_style(ax, tick_fs=RATIO_TICK_FS)
    save_fig(fig, outpath)

def plot_enzh_triad(curves_path: str, outpath: str):
    df = pd.read_csv(curves_path)

    # Expecting: pair, doc_mix, method, mix_ratio, ndcg10, ...
    tri = df[(df["pair"] == "EN-ZH") & (df["method"] == "embed")].copy()
    want = ["EN docs", "ZH docs", "EN + ZH docs"]
    tri = tri[tri["doc_mix"].isin(want)]

    fig, ax = plt.subplots(figsize=RATIO_FIGSIZE)
    markers = ["o", "s", "^"]
    for color, doc_mix, marker in zip(PALETTE, want, markers):
        sub = tri[tri["doc_mix"] == doc_mix].sort_values("mix_ratio")
        ax.plot(
            sub["mix_ratio"],
            sub["ndcg10"],
            marker=marker,
            linewidth=RATIO_LINE_WIDTH,
            markersize=RATIO_MARKER_SIZE,
            color=color,
            label=doc_mix,
        )

    ax.set_xlabel("Mix ratio (% ZH)", fontsize=RATIO_LABEL_FS)
    ax.set_ylabel("nDCG@10", fontsize=RATIO_LABEL_FS)
    ax.set_title("EN–ZH nDCG@10 vs mix ratio", fontsize=RATIO_TITLE_FS)
    apply_academic_style(ax, tick_fs=RATIO_TICK_FS)
    ax.legend(frameon=False, loc="upper right", fontsize=RATIO_LEGEND_FS)
    save_fig(fig, outpath)

def plot_hub_examples(df: pd.DataFrame, outpath: str, doc_langs=("DE","ES","FR","NL","ZH")):
    # Monolingual docs only
    mono = df[df["doc_type"] == "mono"].copy()
    mono["doc_lang"] = mono["doc_mix"].str.replace(" docs", "", regex=False)
    mono["doc_lang_lc"] = mono["doc_lang"].str.lower()

    # partner language = other side of the pair
    mono["partner_lc"] = np.where(
        mono["lang_a"] == mono["doc_lang_lc"], mono["lang_b"], mono["lang_a"]
    )
    mono["partner"] = mono["partner_lc"].str.upper()

    rows = []
    for L in doc_langs:
        sub = mono[mono["doc_lang"] == L].copy()
        if sub.empty:
            continue

        # Δ with EN as partner
        en_row = sub[sub["partner"] == "EN"]
        if en_row.empty:
            continue
        delta_en = float(en_row["delta_ndcg"].iloc[0])

        # Top-2 non-EN partners
        sub_non = sub[sub["partner"] != "EN"].sort_values("delta_ndcg", ascending=False)
        best1_partner = str(sub_non["partner"].iloc[0]) if len(sub_non) >= 1 else ""
        best1_delta = float(sub_non["delta_ndcg"].iloc[0]) if len(sub_non) >= 1 else np.nan
        best2_partner = str(sub_non["partner"].iloc[1]) if len(sub_non) >= 2 else ""
        best2_delta = float(sub_non["delta_ndcg"].iloc[1]) if len(sub_non) >= 2 else np.nan

        rows.append((L, delta_en, best1_partner, best1_delta, best2_partner, best2_delta))

    res = pd.DataFrame(
        rows,
        columns=[
            "doc_lang",
            "delta_EN",
            "best1_partner",
            "delta_best1_nonEN",
            "best2_partner",
            "delta_best2_nonEN",
        ],
    )

    if res.empty:
        return

    x = np.arange(len(res))
    w = 0.26

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=2)

    bars_en = ax.bar(
        x - w,
        res["delta_EN"],
        width=w,
        color=PALETTE[0],
        alpha=0.8,
        edgecolor=EDGE_COLOR,
    )
    bars_best1 = ax.bar(
        x,
        res["delta_best1_nonEN"],
        width=w,
        color=PALETTE[1],
        alpha=0.8,
        edgecolor=EDGE_COLOR,
    )
    bars_best2 = ax.bar(
        x + w,
        res["delta_best2_nonEN"],
        width=w,
        color=PALETTE[2],
        alpha=0.8,
        edgecolor=EDGE_COLOR,
    )

    # Label each bar with the partner language it represents (removes need for a legend).
    max_abs = float(
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        res["delta_EN"].to_numpy(),
                        res["delta_best1_nonEN"].to_numpy(),
                        res["delta_best2_nonEN"].to_numpy(),
                    ]
                )
            )
        )
    )
    partner_label_offset = max(0.002, 0.05 * max_abs) if max_abs > 0 else 0.002

    def label_bars(bars, labels):
        for bar, lab in zip(bars, labels):
            if not lab:
                continue
            h = float(bar.get_height())
            if np.isnan(h):
                continue
            xc = bar.get_x() + bar.get_width() / 2
            if abs(h) >= 0.02:
                ax.text(
                    xc,
                    h / 2,
                    lab,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                )
            else:
                y = h + (partner_label_offset if h >= 0 else -partner_label_offset)
                ax.text(
                    xc,
                    y,
                    lab,
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=11,
                    fontweight="bold",
                    color=EDGE_COLOR,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7},
                )

    label_bars(bars_en, ["EN"] * len(res))
    label_bars(bars_best1, res["best1_partner"].tolist())
    label_bars(bars_best2, res["best2_partner"].tolist())

    ax.set_xticks(x, res["doc_lang"])
    ax.set_xlabel("Document Language", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\Delta$ nDCG@10", fontsize=LABEL_FS)
    ax.set_title("Partner effect: EN vs top-2 non-EN", fontsize=TITLE_FS)
    apply_academic_style(ax)
    save_fig(fig, outpath)

def plot_typology_scatter(df: pd.DataFrame, outpath: str):
    mono = df[df["doc_type"] == "mono"].copy()
    sub = mono[(mono["lang_a"] != "en") & (mono["lang_b"] != "en")].copy()

    x = sub["lang2vec_knn"].to_numpy()
    y = sub["delta_ndcg"].to_numpy()

    rho, p = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=2)
    ax.scatter(x, y, s=SCATTER_POINT_SIZE, alpha=0.7, color=PALETTE[0])

    # Simple least-squares trend line (optional but helpful visually)
    if len(x) >= 2:
        m, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, m*xs + c, color=EDGE_COLOR, linewidth=1.6)

    ax.set_xlabel("Typological distance (lang2vec_knn)", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\Delta$ nDCG@10", fontsize=LABEL_FS)
    ax.set_title(f"Non-EN monolingual docs: ρ={rho:.3f}, n={len(sub)}", fontsize=TITLE_FS)
    apply_academic_style(ax)
    save_fig(fig, outpath)

def plot_headroom_scatter(df: pd.DataFrame, outpath: str):
    sub = df.copy()
    if "doc_type" in sub.columns:
        sub["doc_type"] = sub["doc_type"].fillna("unknown").str.lower()
    else:
        sub["doc_type"] = "mono"

    color_map = {False: PALETTE[0], True: PALETTE[1]}
    marker_map = {"mono": "o", "bi": "s"}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=2)

    for doc_type, marker in marker_map.items():
        sub_doc = sub[sub["doc_type"] == doc_type]
        for en_in_index, color in color_map.items():
            sub_en = sub_doc[sub_doc["en_in_index"] == en_in_index]
            if sub_en.empty:
                continue
            ax.scatter(
                sub_en["best_endpoint_ndcg"],
                sub_en["delta_ndcg"],
                s=HEADROOM_POINT_SIZE,
                alpha=0.7,
                color=color,
                marker=marker,
                edgecolor="none",
            )

    ax.set_xlabel("Best endpoint nDCG@10", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\Delta$ nDCG@10", fontsize=LABEL_FS)
    ax.set_title("Headroom scatter", fontsize=TITLE_FS)
    apply_academic_style(ax)

    def wrap_legend_label(text, width=16):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=PALETTE[0],
            markeredgecolor="none",
            label=wrap_legend_label("Mono non-EN docs"),
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor=PALETTE[0],
            markeredgecolor="none",
            label=wrap_legend_label("Bilingual docs without EN"),
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=PALETTE[1],
            markeredgecolor="none",
            label=wrap_legend_label("Mono EN docs"),
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor=PALETTE[1],
            markeredgecolor="none",
            label=wrap_legend_label("Bilingual docs with EN"),
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=LEGEND_FS,
        loc="upper right",
        ncol=1,
        handletextpad=0.5,
        borderpad=0.3,
        labelspacing=0.4,
    )
    save_fig(fig, outpath)

def plot_mono_alignment_curve(curves_path: str, outpath: str):
    df = pd.read_csv(curves_path)

    mono = df[~df["doc_mix"].str.contains(r"\+", na=False)].copy()
    mono["doc_lang"] = mono["doc_mix"].str.replace(" docs", "", regex=False).str.strip().str.upper()
    mono["mix_ratio"] = pd.to_numeric(mono["mix_ratio"], errors="coerce")

    pair_split = mono["pair"].str.replace("–", "-", regex=False).str.split("-", n=1, expand=True)
    mono["lang_a"] = pair_split[0].str.strip().str.upper()
    mono["lang_b"] = pair_split[1].str.strip().str.upper()

    lam = mono["mix_ratio"]
    mono["p_doc"] = np.where(
        mono["doc_lang"] == mono["lang_a"],
        100.0 - lam,
        np.where(mono["doc_lang"] == mono["lang_b"], lam, np.nan),
    )

    sub = mono.dropna(subset=["p_doc", "ndcg10"]).copy()
    sub["p_doc"] = sub["p_doc"].round().astype(int)

    stats = (
        sub.groupby("p_doc", as_index=False)["ndcg10"]
        .agg(mean="mean", std="std", count="count")
        .sort_values("p_doc")
    )
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    yerr = np.nan_to_num(stats["stderr"].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        stats["p_doc"],
        stats["mean"],
        yerr=yerr,
        marker="o",
        linewidth=LINE_PLOT_WIDTH,
        markersize=LINE_PLOT_MARKER_SIZE,
        color=PALETTE[0],
        ecolor=EDGE_COLOR,
        capsize=2,
        label="Mean nDCG@10",
    )

    ax.set_xlabel("p_doc (query share in doc language, %)", fontsize=LABEL_FS)
    ax.set_ylabel("nDCG@10", fontsize=LABEL_FS)
    ax.set_title("Monolingual alignment curve", fontsize=TITLE_FS)
    ax.set_xlim(-2, 102)
    ax.set_xticks(np.arange(0, 101, 10))
    apply_academic_style(ax)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="best")
    save_fig(fig, outpath)

def parse_pair_langs(pair: str):
    pair = pair.replace("–", "-")
    parts = pair.split("-", 1)
    lang_a = parts[0].strip().upper()
    lang_b = parts[1].strip().upper() if len(parts) > 1 else ""
    return lang_a, lang_b

def plot_lambda_star_summary(curves_path: str, outpath: str):
    df = pd.read_csv(curves_path)
    df = df[df["method"] == "embed"].copy()
    df["mix_ratio"] = pd.to_numeric(df["mix_ratio"], errors="coerce")
    df = df.dropna(subset=["mix_ratio", "ndcg10"])

    records = []
    for (pair, doc_mix), group in df.groupby(["pair", "doc_mix"]):
        group = group.sort_values("mix_ratio")
        idx = group["ndcg10"].idxmax()
        row = group.loc[idx]
        mix_ratio = float(row["mix_ratio"])
        lang_a, lang_b = parse_pair_langs(pair)
        en_pair = lang_a == "EN" or lang_b == "EN"
        is_bi = "+" in doc_mix

        rec = {
            "pair": pair,
            "doc_mix": doc_mix,
            "en_pair": en_pair,
            "is_bi": is_bi,
        }

        lam = int(round(mix_ratio))
        if is_bi:
            rec["lambda_star"] = lam
        else:
            doc_lang = doc_mix.replace(" docs", "").strip().upper()
            if doc_lang == lang_a:
                p_doc = 100.0 - lam
            elif doc_lang == lang_b:
                p_doc = lam
            else:
                p_doc = None
            rec["doc_lang"] = doc_lang
            rec["p_doc"] = p_doc
        records.append(rec)

    mono_counts = defaultdict(lambda: defaultdict(int))
    bi_counts = defaultdict(lambda: defaultdict(int))

    for rec in records:
        if rec["is_bi"]:
            subset = "EN bilingual" if rec["en_pair"] else "Non-EN bilingual"
            lam = rec.get("lambda_star")
            if lam is None:
                continue
            key = int(lam)
            bi_counts[subset][key] += 1
        else:
            if not rec["en_pair"]:
                subset = "Non-EN monolingual"
            else:
                subset = "EN monolingual (EN docs)" if rec.get("doc_lang") == "EN" else "EN monolingual (non-EN docs)"

            p_doc = rec.get("p_doc")
            if p_doc is None:
                continue
            key = int(p_doc)
            mono_counts[subset][key] += 1

    def shade_color(color, strength):
        rgb = np.array(mcolors.to_rgb(color))
        if strength >= 0:
            return rgb * (1 - strength) + np.ones(3) * strength
        strength = -strength
        return rgb * (1 - strength)

    def pick_colors(base_color: str, n: int):
        if n <= 1:
            return [base_color]
        strengths = np.linspace(-0.15, 0.45, n)
        return [mcolors.to_hex(shade_color(base_color, s)) for s in strengths]

    def text_color_for(bg_color):
        r, g, b, _ = mcolors.to_rgba(bg_color)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.6 else "white"

    def wrap_label(text, width=16):
        lines = []
        for chunk in text.split("\n"):
            wrapped = textwrap.wrap(chunk.strip(), width=width, break_long_words=False)
            lines.extend(wrapped if wrapped else [""])
        return "\n".join(lines)

    def plot_group(
        ax,
        subset_order,
        subset_labels,
        counts_by_subset,
        categories,
        colors,
        label_func,
        x_start,
        bar_width,
        bar_step,
        label_width,
    ):
        totals = {k: sum(counts_by_subset.get(k, {}).values()) for k in subset_order}
        x = np.arange(len(subset_order)) * bar_step + x_start
        bottoms = np.zeros(len(subset_order))
        min_label_share = 8.0

        for cat, color in zip(categories, colors):
            vals = [counts_by_subset.get(k, {}).get(cat, 0) for k in subset_order]
            shares = [100.0 * v / totals[k] if totals[k] else 0.0 for v, k in zip(vals, subset_order)]
            bars = ax.bar(
                x,
                shares,
                bottom=bottoms,
                width=bar_width,
                color=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
            )
            for i, share in enumerate(shares):
                if share >= min_label_share:
                    ax.text(
                        x[i],
                        bottoms[i] + share / 2.0,
                        label_func(cat),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=text_color_for(color),
                        fontweight="bold",
                    )
            bottoms += np.array(shares)

        labels = [f"{wrap_label(subset_labels[k], width=label_width)}\n(n={totals[k]})" for k in subset_order]
        return list(x), labels

    mono_order = [
        "Non-EN monolingual",
        "EN monolingual (EN docs)",
        "EN monolingual (non-EN docs)",
    ]
    mono_labels = {
        "Non-EN monolingual": "Non-EN pairs \nmonolingual docs",
        "EN monolingual (EN docs)": "EN pairs \nmonolingual EN docs",
        "EN monolingual (non-EN docs)": "EN pairs \nmonolingual non-EN docs",
    }
    mono_cats = sorted({k for v in mono_counts.values() for k in v.keys()})

    bi_order = ["Non-EN bilingual", "EN bilingual"]
    bi_labels = {
        "Non-EN bilingual": "Non-EN pairs \nbilingual docs",
        "EN bilingual": "EN pairs \nbilingual docs",
    }
    bi_cats = sorted({k for v in bi_counts.values() for k in v.keys()})

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x_positions = []
    x_labels = []

    bar_width = 0.5
    bar_step = 0.6
    group_gap = 0.12
    label_width = 12
    mono_x, mono_lbls = plot_group(
        ax,
        mono_order,
        mono_labels,
        mono_counts,
        mono_cats,
        pick_colors(PALETTE[0], max(len(mono_cats), 1)),
        lambda v: rf"$p_{{\mathrm{{doc}}}}={v:d}\%$",
        x_start=0,
        bar_width=bar_width,
        bar_step=bar_step,
        label_width=label_width,
    )
    x_positions += mono_x
    x_labels += mono_lbls

    bi_start = mono_x[-1] + bar_step + group_gap if mono_x else 0
    bi_x, bi_lbls = plot_group(
        ax,
        bi_order,
        bi_labels,
        bi_counts,
        bi_cats,
        pick_colors(PALETTE[1], max(len(bi_cats), 1)),
        lambda v: rf"$\lambda^*={v:d}\%$",
        x_start=bi_start,
        bar_width=bar_width,
        bar_step=bar_step,
        label_width=label_width,
    )
    x_positions += bi_x
    x_labels += bi_lbls

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0, 101, 20))
    pad = bar_width / 2.0 + 0.15
    ax.set_xlim(min(x_positions) - pad, max(x_positions) + pad)
    ax.set_title("Distribution of optimal mixing ratios", fontsize=TITLE_FS)
    ax.set_ylabel("Share of settings (%)", fontsize=LABEL_FS)
    apply_academic_style(ax)
    ax.tick_params(axis="x", labelsize=TICK_FS - 2)
    for label in ax.get_xticklabels():
        label.set_multialignment("center")
    save_fig(fig, outpath)

def pick_ratio_source(primary: str, fallback: str):
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    return None

LANG_MAP = {
    "english": "EN",
    "en": "EN",
    "chinese": "ZH",
    "zh": "ZH",
    "indonesian": "ID",
    "id": "ID",
}

def normalize_doc_mix(val, row):
    v = "" if not isinstance(val, str) else val.strip()
    y = v.lower().replace("  ", " ")
    if any(sym in y for sym in ["%", "&", "+", " and "]):
        y2 = y.replace("%", "&")
        parts = re.split(r"\s*(?:&|\+|and)\s*", y2.replace(" docs", "").replace("documents", ""))
        parts = [LANG_MAP.get(p.strip(), p.strip().upper()) for p in parts if p.strip()]
        langs = [p for p in parts if p in {"EN", "ZH", "ID"}]
        if len(langs) >= 2:
            return f"{langs[0]} & {langs[1]} docs"
    if any(k in y for k in ["bilingual", "mixed", "dual-lang", "two-language"]):
        l1 = LANG_MAP.get(str(row.get("q_lang_1", "")).lower(), str(row.get("q_lang_1", "")).upper())
        l2 = LANG_MAP.get(str(row.get("q_lang_2", "")).lower(), str(row.get("q_lang_2", "")).upper())
        if l1 and l2:
            return f"{l1} & {l2} docs"
        return "Bilingual docs"
    for k, abbr in [
        ("english", "EN"),
        ("en docs", "EN"),
        ("chinese", "ZH"),
        ("zh docs", "ZH"),
        ("indonesian", "ID"),
        ("id docs", "ID"),
    ]:
        if k in y or y == abbr.lower():
            return f"{abbr} docs"
    return v if "docs" in v else (v + " docs")

def load_ratio_curve_data(primary: str, fallback: str):
    path = pick_ratio_source(primary, fallback)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "mean" not in df.columns:
        if "value" in df.columns:
            df["mean"] = df["value"]
        else:
            return None, path
    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.lower()
    if "mix_ratio" not in df.columns:
        return None, path
    df["mix_ratio"] = pd.to_numeric(df["mix_ratio"], errors="coerce")
    if "pair" in df.columns:
        df["pair"] = (
            df["pair"]
            .astype(str)
            .str.upper()
            .str.replace("–", "-", regex=False)
            .str.replace("—", "-", regex=False)
        )
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.strip().str.lower()
    else:
        df["method"] = "method"
    if "doc_mix" not in df.columns:
        df["doc_mix"] = "docs"
    else:
        if "q_lang_1" not in df.columns:
            df["q_lang_1"] = ""
        if "q_lang_2" not in df.columns:
            df["q_lang_2"] = ""
        df["doc_mix"] = [
            normalize_doc_mix(v, r)
            for v, r in zip(df["doc_mix"], df.to_dict(orient="records"))
        ]
    df_fig = df[df["metric"] == "ndcg@10"].copy() if "metric" in df.columns else df.copy()
    return df_fig, path

def drop_dupes_for_plot(sub):
    sub = sub.dropna(subset=["mix_ratio", "mean"]).copy()
    if sub.empty:
        return sub
    sub["method"] = sub["method"].astype(str).str.strip().str.lower()
    return sub.groupby(["method", "mix_ratio"], as_index=False).agg(mean=("mean", "mean"))

def choose_xytext(ax, x, y, dx, dy):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad_x = 0.01 * (xmax - xmin)
    pad_y = 0.01 * (ymax - ymin)

    candidates = [
        (x + dx, y + dy, "left", "bottom"),
        (x - dx, y + dy, "right", "bottom"),
        (x + dx, y - dy, "left", "top"),
        (x - dx, y - dy, "right", "top"),
    ]
    for xx, yy, ha, va in candidates:
        if (xmin + pad_x) <= xx <= (xmax - pad_x) and (ymin + pad_y) <= yy <= (ymax - pad_y):
            return (xx, yy, ha, va)

    xx = min(max(x, xmin + pad_x), xmax - pad_x)
    yy = min(max(y, ymin + pad_y), ymax - pad_y)
    return (xx, yy, "center", "center")

def slugify(text):
    text = text.replace("&", "and").replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9]+", "", text)

def plot_ratio_curve(sub, title, outpath, scale_y=1.0):
    is_enzh = False
    if "pair" in sub.columns:
        pairs = [str(p).replace("–", "-").strip().upper() for p in sub["pair"].dropna().unique()]
        if len(pairs) == 1 and pairs[0].replace("-", "") == "ENZH":
            is_enzh = True

    sub = drop_dupes_for_plot(sub)
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=RATIO_FIGSIZE)

    methods = sorted(sub["method"].dropna().unique())
    extra_colors = [mcolors.to_hex(c) for c in plt.get_cmap("tab10").colors]
    color_cycle = PALETTE + [c for c in extra_colors if c not in PALETTE]
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", ">", "<"]

    for i, method in enumerate(methods):
        g = sub[sub["method"] == method].sort_values("mix_ratio")
        color = color_cycle[i % len(color_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        y_vals = g["mean"] * scale_y
        ax.plot(
            g["mix_ratio"],
            y_vals,
            marker=marker,
            markersize=RATIO_MARKER_SIZE,
            linewidth=RATIO_LINE_WIDTH,
            color=color,
            label=str(method),
        )

    idx = sub["mean"].idxmax()
    r = float(sub.loc[idx, "mix_ratio"])
    v = float(sub.loc[idx, "mean"]) * scale_y
    ax.scatter([r], [v], s=RATIO_HIGHLIGHT_SIZE ** 2, color=PALETTE[1], zorder=5)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = RATIO_ANNOT_DX_FRAC * (xmax - xmin)
    dy = RATIO_ANNOT_DY_FRAC * (ymax - ymin)
    xx, yy, ha, va = choose_xytext(ax, r, v, dx, dy)
    ax.annotate(
        f"{v:.4f} @ {int(round(r))}",
        xy=(r, v),
        xytext=(xx, yy),
        ha=ha,
        va=va,
        fontsize=RATIO_ANNOT_FS,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=RATIO_ANNOT_BBOX_ALPHA),
        arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6),
    )

    present = sorted(set(int(x) for x in sub["mix_ratio"].dropna().round().astype(int)))
    xticks = [t for t in RATIO_XTICKS if t in present] or present
    ax.set_xticks(xticks)
    ax.set_xlabel("Mix ratio (% ZH)" if is_enzh else "Mix ratio (% target language)", fontsize=RATIO_LABEL_FS)
    ylabel = "nDCG@10 (×100)" if scale_y != 1.0 else "nDCG@10"
    ax.set_ylabel(ylabel, fontsize=RATIO_LABEL_FS)
    ax.set_title(title, fontsize=RATIO_TITLE_FS)
    apply_academic_style(ax)
    ax.tick_params(axis="both", labelsize=RATIO_TICK_FS)
    ax.legend(
        frameon=False,
        fontsize=RATIO_LEGEND_FS,
        title=RATIO_LEGEND_TITLE,
        title_fontsize=RATIO_LEGEND_TITLE_FS,
        loc="best",
    )
    save_fig(fig, outpath)

def plot_ratio_curves(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for pair in sorted(df["pair"].dropna().unique()):
        sub_pair = df[df["pair"] == pair]
        for docmix in sorted(sub_pair["doc_mix"].dropna().unique()):
            sub = sub_pair[sub_pair["doc_mix"] == docmix]
            if sub.empty:
                continue
            title = f"{pair} - {docmix}"
            fname = f"ratio_curve_{slugify(pair)}_{slugify(docmix)}.pdf"
            pair_key = str(pair).replace("–", "-").replace("-", "").upper()
            scale_y = 100.0 if pair_key == "ENZH" else 1.0
            plot_ratio_curve(sub, title, os.path.join(outdir, fname), scale_y=scale_y)

def band_midpoint(band_str: str) -> float:
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", str(band_str))
    if m:
        a, b = map(float, m.groups())
        return (a + b) / 2.0
    try:
        return float(band_str)
    except ValueError:
        return np.nan

def add_linear_fit_with_r2(ax, xs, ys, color=EDGE_COLOR):
    if len(xs) < 2:
        return np.nan
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    coef = np.polyfit(xs, ys, 1)
    fit_y = np.polyval(coef, xs)
    ss_res = np.sum((ys - fit_y) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ax.plot(
        xs,
        fit_y,
        linestyle="-.",
        linewidth=1.8,
        color=color,
        alpha=0.7,
        label=f"linear fit (R^2={r2:.3f})",
    )
    return r2

def plot_projection_panel(ax, x, series, title, xlabel, ylabel):
    for label, y, color, marker, linestyle in series:
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=LINE_PLOT_WIDTH,
            markersize=LINE_PLOT_MARKER_SIZE,
        )
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    apply_academic_style(ax)
    if series:
        ax.legend(frameon=False, fontsize=LEGEND_FS, loc="best")

def pick_embedding_bands_path(primary: str, fallbacks):
    if os.path.exists(primary):
        return primary
    candidates = [path for path in fallbacks if os.path.exists(path)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=os.path.getmtime)

def plot_embedding_projections(bands_csv: str, outdir: str):
    df = pd.read_csv(bands_csv)
    col_map = {c.lower(): c for c in df.columns}
    band_col = col_map.get("band")
    midpoint_col = col_map.get("mix_midpoint_est")
    if band_col is None and midpoint_col is None:
        print(f"Skipping embedding projections; missing band/mix_midpoint_est in {bands_csv}")
        return
    if midpoint_col is not None:
        mid_vals = pd.to_numeric(df[midpoint_col], errors="coerce")
        scale = 100.0 if mid_vals.max(skipna=True) <= 1.5 else 1.0
        df["band_mid"] = mid_vals * scale
    else:
        df["band_mid"] = df[band_col].apply(band_midpoint)
    df = df.dropna(subset=["band_mid"]).sort_values("band_mid")
    x = df["band_mid"].to_numpy()

    def get_series(*keys):
        col = None
        for key in keys:
            col = col_map.get(key)
            if col is not None:
                break
        if col is None:
            return None
        return pd.to_numeric(df[col], errors="coerce").to_numpy()

    r_mean = get_series("r_mean", "mean_r")
    r_median = get_series("r_median", "median_r")
    r_trimmed = get_series("r_trimmed", "trimmed_mean_r_5pct", "trimmed_mean_r")
    delta_mean = get_series("delta_mean", "mean_delta")
    delta_median = get_series("delta_median", "median_delta")
    delta_trimmed = get_series("delta_trimmed", "trimmed_mean_delta_5pct", "trimmed_mean_delta")
    if all(v is None for v in [r_mean, r_median, r_trimmed, delta_mean, delta_median, delta_trimmed]):
        print(f"Skipping embedding projections; missing r/delta columns in {bands_csv}")
        return

    out_dir = os.path.join(outdir, "embedding_projections")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    r_series = []
    if r_mean is not None:
        r_series.append(("r (mean)", r_mean, PALETTE[0], "o", "-"))
    if r_median is not None:
        r_series.append(("r (median)", r_median, PALETTE[1], "^", "--"))
    if r_trimmed is not None:
        r_series.append(("r (trimmed)", r_trimmed, PALETTE[2], "s", ":"))
    r_series = [s for s in r_series if np.any(np.isfinite(s[1]))]
    plot_projection_panel(
        ax,
        x,
        r_series,
        "Position along EN-ZH axis (r)",
        "Mixing band midpoint (% ZH)",
        "r (0 = EN end, 1 = ZH end)",
    )
    if r_mean is not None and np.sum(np.isfinite(r_mean)) >= 2:
        add_linear_fit_with_r2(ax, x, r_mean, color=EDGE_COLOR)
        ax.legend(frameon=False, fontsize=LEGEND_FS, loc="best")
    save_fig(fig, os.path.join(out_dir, "r_position.pdf"))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    d_series = []
    if delta_mean is not None:
        d_series.append((r"$\delta$ (mean)", delta_mean, PALETTE[0], "o", "-"))
    if delta_median is not None:
        d_series.append((r"$\delta$ (median)", delta_median, PALETTE[1], "^", "--"))
    if delta_trimmed is not None:
        d_series.append((r"$\delta$ (trimmed)", delta_trimmed, PALETTE[2], "s", ":"))
    d_series = [s for s in d_series if np.any(np.isfinite(s[1]))]
    plot_projection_panel(
        ax,
        x,
        d_series,
        "Distance to EN-ZH axis (delta)",
        "Mixing band midpoint (% ZH)",
        r"$\delta$ (smaller = closer to EN-ZH)",
    )
    save_fig(fig, os.path.join(out_dir, "delta_offset.pdf"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2.1, FIGSIZE[1]))
    plot_projection_panel(
        ax1,
        x,
        r_series,
        "Position along EN-ZH axis (r)",
        "% ZH",
        "r",
    )
    if r_mean is not None and np.sum(np.isfinite(r_mean)) >= 2:
        add_linear_fit_with_r2(ax1, x, r_mean, color=EDGE_COLOR)
        ax1.legend(frameon=False, fontsize=LEGEND_SMALL_FS, loc="best")
    plot_projection_panel(
        ax2,
        x,
        d_series,
        "Distance to axis (δ)",
        "% ZH",
        r"$\delta$",
    )
    if d_series:
        ax2.legend(frameon=False, fontsize=LEGEND_SMALL_FS, loc="best")
    save_fig(fig, os.path.join(out_dir, "embedding_projections_2col.pdf"))

def pick_row(df: pd.DataFrame, model: str, pair: str, doc_regime: str, preferred_blocks=None):
    sub = df[(df["model"] == model) & (df["pair"] == pair) & (df["doc_regime"] == doc_regime)]
    if preferred_blocks is not None:
        for b in preferred_blocks:
            sub_b = sub[sub["block"] == b]
            if len(sub_b) > 0:
                return sub_b.iloc[0]
    if len(sub) == 0:
        return None
    return sub.iloc[0]

def dumbbell_plot(models, left_label, right_label, left_vals, right_vals, outpath,
                  xlabel=r"$\Delta$ (nDCG@10)"):
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.axvline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=2)

    # connectors
    for i in range(len(models)):
        ax.plot([left_vals[i], right_vals[i]], [y[i], y[i]], linewidth=DUMBBELL_LINE_WIDTH, color=EDGE_COLOR)

    # points (use different markers, no manual colors)
    ax.plot(left_vals, y, linestyle="None", marker="s", markersize=LINE_PLOT_MARKER_SIZE, label=left_label)
    ax.plot(right_vals, y, linestyle="None", marker="o", markersize=LINE_PLOT_MARKER_SIZE, label=right_label)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=TICK_FS)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    apply_academic_style(ax)
    ax.legend(frameon=False, loc="best", fontsize=LEGEND_FS)
    ax.invert_yaxis()
    save_fig(fig, outpath)

def plot_ablation_hub(df: pd.DataFrame, outdir: str):
    # Hub ablation (model family)
    family_models = [
        "multilingual-e5-large-instruct",
        "gte-multilingual-base",
        "jina-embeddings-v3",
        "Qwen3-Embedding-0.6B",
    ]

    # ZH docs: EN-ZH vs ID-ZH
    enzh = []
    idzh = []
    for m in family_models:
        r1 = pick_row(df, m, "EN-ZH", "L2 docs", preferred_blocks=["composition", "size"])
        r2 = pick_row(df, m, "ID-ZH", "L2 docs", preferred_blocks=["hub", "size"])
        assert r1 is not None and r2 is not None, f"Missing ZH-doc hub row for {m}"
        enzh.append(float(r1["delta_ndcg"]))
        idzh.append(float(r2["delta_ndcg"]))

    dumbbell_plot(
        family_models,
        left_label="ID–ZH on ZH docs",
        right_label="EN–ZH on ZH docs",
        left_vals=idzh,
        right_vals=enzh,
        outpath=os.path.join(outdir, "ablation_hub_ZH.pdf"),
    )

    # DE docs: DE-EN vs DE-NL
    deen = []
    denl = []
    for m in family_models:
        r1 = pick_row(df, m, "DE-EN", "L1 docs", preferred_blocks=["hub"])
        r2 = pick_row(df, m, "DE-NL", "L1 docs", preferred_blocks=["composition", "size"])
        assert r1 is not None and r2 is not None, f"Missing DE-doc hub row for {m}"
        deen.append(float(r1["delta_ndcg"]))
        denl.append(float(r2["delta_ndcg"]))

    dumbbell_plot(
        family_models,
        left_label="DE–NL on DE docs",
        right_label="DE–EN on DE docs",
        left_vals=denl,
        right_vals=deen,
        outpath=os.path.join(outdir, "ablation_hub_DE.pdf"),
    )

def plot_qwen_scale(df: pd.DataFrame, outpath: str):
    # Qwen scale plot
    qwen_sizes = ["0.6B", "4B", "8B"]
    qwen_models = [f"Qwen3-Embedding-{s}" for s in qwen_sizes]

    # settings: (pair, doc_regime, label)
    scale_settings = [
        ("EN-ZH", "L1 docs", "EN–ZH on EN docs (EN in index)"),
        ("EN-ZH", "L2 docs", "EN–ZH on ZH docs (EN absent)"),
        ("ID-ZH", "L2 docs", "ID–ZH on ZH docs (non-EN pair)"),
        ("DE-NL", "L1 docs", "DE–NL on DE docs (non-EN pair)"),
        ("EN-AR", "L2 docs", "EN–AR on AR docs (EN absent)"),
    ]

    def size_to_x(s):
        # numeric x for plotting; preserves ordering
        return {"0.6B": 0.6, "4B": 4.0, "8B": 8.0}[s]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axhline(0.0, color=LINE_ZERO_COLOR, linewidth=LINE_ZERO_WIDTH, zorder=2)

    line_markers = ["o", "s", "^", "D", "P"]
    for (pair, reg, label), marker in zip(scale_settings, line_markers):
        xs, ys = [], []
        for s, m in zip(qwen_sizes, qwen_models):
            r = pick_row(df, m, pair, reg, preferred_blocks=["size", "composition", "hub"])
            if r is None:
                continue
            xs.append(size_to_x(s))
            ys.append(float(r["delta_ndcg"]))
        if len(xs) >= 2:
            # line + markers; no manual colors
            ax.plot(xs, ys, marker=marker, linewidth=LINE_PLOT_WIDTH, markersize=LINE_PLOT_MARKER_SIZE, label=label)
        elif len(xs) == 1:
            ax.plot(xs, ys, marker=marker, linestyle="None", markersize=LINE_PLOT_MARKER_SIZE, label=label)

    ax.set_xlabel("Model size (B parameters; Qwen3 Embedding)", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\Delta$ (nDCG@10)", fontsize=LABEL_FS)
    apply_academic_style(ax)
    ax.legend(frameon=False, fontsize=LEGEND_SMALL_FS, loc="best")
    save_fig(fig, outpath)

def main():
    df = load_and_filter_processed(PROCESSED)

    plot_delta_distribution(df, os.path.join(OUTDIR, "delta_distribution_all.pdf"))
    plot_en_in_index_split(df, os.path.join(OUTDIR, "en_in_index_split.pdf"))
    plot_enzh_triad(CURVES, os.path.join(OUTDIR, "triad_ENZH.pdf"))
    plot_hub_examples(df, os.path.join(OUTDIR, "hub_sweeps.pdf"))
    plot_typology_scatter(df, os.path.join(OUTDIR, "typology_scatter.pdf"))
    plot_headroom_scatter(df, os.path.join(OUTDIR, "headroom_scatter.pdf"))
    plot_mono_alignment_curve(CURVES, os.path.join(OUTDIR, "mono_alignment_curve.pdf"))
    plot_lambda_star_summary(CURVES, os.path.join(OUTDIR, "lambda_star_summary.pdf"))

    ratio_df, _ = load_ratio_curve_data(RATIO_PRIMARY, RATIO_FALLBACK)
    if ratio_df is not None:
        plot_ratio_curves(ratio_df, os.path.join(OUTDIR, "ratio_curves"))
    else:
        print("Skipping ratio curves (no unified/converted results CSV found).")

    bands_path = pick_embedding_bands_path(EMBEDDING_BANDS, EMBEDDING_BANDS_FALLBACKS)
    if bands_path is not None:
        plot_embedding_projections(bands_path, OUTDIR)
    else:
        print("Skipping embedding projections (missing embedding_bands/band_summaries CSV).")

    ablation_df = pd.read_csv(ABLATION)
    plot_ablation_hub(ablation_df, OUTDIR)
    plot_qwen_scale(ablation_df, os.path.join(OUTDIR, "qwen_scale.pdf"))

    print("Done. Wrote PDFs to", OUTDIR)

if __name__ == "__main__":
    main()

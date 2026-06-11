"""Calculate paper table values with readable output."""
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "artifacts" / "tables"
processed_path = str(TABLES_DIR / "full_mmarco_processed_results.csv")
pivot_path = str(TABLES_DIR / "full_mmarco_results.csv")
PERMUTATIONS = 5000
BOOTSTRAPS = 10000

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def is_endpoint(x):
    try:
        v = float(x)
    except Exception:
        return False
    return abs(v - 0.0) < 1e-9 or abs(v - 100.0) < 1e-9

def parse_doc_langs(doc_mix):
    text = doc_mix.replace("docs", "").replace("+", " ")
    parts = [p.strip() for p in text.split() if p.strip()]
    return [p.upper() for p in parts]

def doc_lang_from_regime(doc_regime, lang_a, lang_b):
    if doc_regime == "L1 docs":
        return lang_a
    if doc_regime == "L2 docs":
        return lang_b
    return None

def print_section(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")

def median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

def rankdata(values):
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j
    return ranks

def pearson(x, y):
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)

def spearman_rho(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson(rx, ry)

def quantile(sorted_vals, q):
    """Return the q-quantile (0..1) using linear interpolation."""
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

def cluster_bootstrap_stats(
    clusters,
    stat_fn,
    n_boot=BOOTSTRAPS,
    seed=0,
):
    """
    Cluster bootstrap CI for a statistic.

    `clusters`: dict cluster_id -> list of records
    `stat_fn`: function(list_of_records) -> float

    Returns (obs, ci_low, ci_high, n_boot_valid).
    """
    all_records = [r for rs in clusters.values() for r in rs]
    obs = stat_fn(all_records)

    keys = list(clusters.keys())
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0

    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        sampled_keys = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        sampled_records = [r for k in sampled_keys for r in clusters[k]]
        val = stat_fn(sampled_records)
        if not math.isnan(val):
            samples.append(val)

    samples.sort()
    ci_low = quantile(samples, 0.025)
    ci_high = quantile(samples, 0.975)
    return obs, ci_low, ci_high, len(samples)

def eta_squared(groups):
    """One-way ANOVA eta^2 = SS_between / SS_total."""
    all_vals = [v for vals in groups.values() for v in vals]
    if len(groups) < 2 or len(all_vals) < 2:
        return float("nan")
    overall = mean(all_vals)
    ss_between = 0.0
    ss_total = sum((v - overall) ** 2 for v in all_vals)
    if ss_total == 0:
        return float("nan")
    for vals in groups.values():
        if not vals:
            continue
        m = mean(vals)
        ss_between += len(vals) * (m - overall) ** 2
    return ss_between / ss_total

def omega_squared(groups):
    """
    One-way ANOVA omega^2 effect size (less biased than eta^2).
    omega^2 = (SS_between - (k-1)*MS_within) / (SS_total + MS_within)
    """
    all_vals = [v for vals in groups.values() for v in vals]
    k = len([g for g, vals in groups.items() if vals])
    if k < 2 or len(all_vals) < 3:
        return float("nan")
    overall = mean(all_vals)
    ss_between = 0.0
    ss_within = 0.0
    ss_total = sum((v - overall) ** 2 for v in all_vals)
    for vals in groups.values():
        if not vals:
            continue
        m = mean(vals)
        ss_between += len(vals) * (m - overall) ** 2
        ss_within += sum((v - m) ** 2 for v in vals)
    df_within = len(all_vals) - k
    if df_within <= 0:
        return float("nan")
    ms_within = ss_within / df_within
    denom = ss_total + ms_within
    if denom == 0:
        return float("nan")
    w2 = (ss_between - (k - 1) * ms_within) / denom
    # omega^2 is conventionally truncated at 0 for interpretability
    return max(0.0, w2)

def perm_spearman(x, y, n_perm=PERMUTATIONS, seed=0):
    obs = spearman_rho(x, y)
    rng = random.Random(seed)
    y_copy = list(y)
    more = 0
    for _ in range(n_perm):
        rng.shuffle(y_copy)
        r = spearman_rho(x, y_copy)
        if abs(r) >= abs(obs):
            more += 1
    p = (more + 1) / (n_perm + 1)
    return obs, p

def perm_mean_diff(groups, a_label=None, b_label=None, n_perm=PERMUTATIONS, seed=0):
    labels = [g for g, vals in groups.items() for _ in vals]
    values = [v for vals in groups.values() for v in vals]
    unique = list(groups.keys())
    if len(unique) != 2:
        return float("nan"), float("nan")
    if a_label is None or b_label is None:
        a_label, b_label = sorted(unique)
    def stat(lbls):
        a_vals = [v for v, l in zip(values, lbls) if l == a_label]
        b_vals = [v for v, l in zip(values, lbls) if l == b_label]
        return mean(a_vals) - mean(b_vals)
    obs = stat(labels)
    rng = random.Random(seed)
    more = 0
    labels_copy = labels[:]
    for _ in range(n_perm):
        rng.shuffle(labels_copy)
        s = stat(labels_copy)
        if abs(s) >= abs(obs):
            more += 1
    p = (more + 1) / (n_perm + 1)
    return obs, p

def anova_f(groups):
    all_vals = [v for vals in groups.values() for v in vals]
    if len(groups) < 2 or not all_vals:
        return float("nan")
    overall = mean(all_vals)
    ss_between = 0.0
    ss_within = 0.0
    for vals in groups.values():
        if not vals:
            continue
        m = mean(vals)
        ss_between += len(vals) * (m - overall) ** 2
        ss_within += sum((v - m) ** 2 for v in vals)
    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)
    if df_within <= 0 or ss_within == 0:
        return float("nan")
    return (ss_between / df_between) / (ss_within / df_within)

def perm_anova(groups, n_perm=PERMUTATIONS, seed=0):
    labels = [g for g, vals in groups.items() for _ in vals]
    values = [v for vals in groups.values() for v in vals]
    obs = anova_f(groups)
    rng = random.Random(seed)
    more = 0
    labels_copy = labels[:]
    for _ in range(n_perm):
        rng.shuffle(labels_copy)
        shuffled = defaultdict(list)
        for v, l in zip(values, labels_copy):
            shuffled[l].append(v)
        f = anova_f(shuffled)
        if f >= obs:
            more += 1
    p = (more + 1) / (n_perm + 1)
    return obs, p

def fmt_mean(val):
    return f"{val:.4f} (norm={val / 100:.4f})"

# --- Valid pairs (need all 3 doc regimes) ---
regs_by_pair = defaultdict(set)
langs_by_pair = {}
processed_rows = []
setting_info = {}
pair_langs = {}

with open(processed_path, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        pair = row["pair"]
        doc_mix = row["doc_mix"]
        doc_regime = row["doc_regime"].strip()
        doc_type = row["doc_type"].strip().lower()
        lang_a = row["lang_a"].strip().lower()
        lang_b = row["lang_b"].strip().lower()
        doc_lang = doc_lang_from_regime(doc_regime, lang_a, lang_b)
        doc_langs = parse_doc_langs(doc_mix)
        en_in_index = "EN" in doc_langs
        record = {
            "pair": pair,
            "doc_mix": doc_mix,
            "doc_regime": doc_regime,
            "doc_type": doc_type,
            "doc_lang": doc_lang,
            "doc_langs": doc_langs,
            "en_in_index": en_in_index,
            "lang_a": lang_a,
            "lang_b": lang_b,
            "best_endpoint_ndcg": to_float(row["best_endpoint_ndcg"]),
            "best_mixed_ndcg": to_float(row["best_mixed_ndcg"]),
            "delta_ndcg": to_float(row["delta_ndcg"]),
            "lambda_star_mid": to_float(row["lambda_star_mid"]),
            "lang2vec_knn": to_float(row["lang2vec_knn"]),
            "glot_tree": to_float(row["glot_tree"]),
            "script_match": row["script_match"].strip().lower(),
            "resource_pattern": row["resource_pattern"].strip(),
        }
        processed_rows.append(record)
        regs_by_pair[pair].add(doc_regime)
        langs_by_pair[pair] = (lang_a, lang_b)
        pair_langs[pair] = (lang_a, lang_b)
        key = (pair, doc_mix)
        if key not in setting_info:
            setting_info[key] = record

valid_pairs = {
    p for p, regs in regs_by_pair.items()
    if {"L1 docs", "L2 docs", "L1+L2 docs"}.issubset(regs)
}
missing_pairs = sorted(p for p in regs_by_pair if p not in valid_pairs)
global_rows = [row for row in processed_rows if row["pair"] in valid_pairs]

# --- Bilingual indexing gains (best_mixed_ndcg) ---
by_pair = defaultdict(dict)
for row in processed_rows:
    pair = row["pair"]
    if pair not in valid_pairs:
        continue
    by_pair[pair][row["doc_regime"]] = row["best_mixed_ndcg"]

non_en = []
en = []
for pair, reg in by_pair.items():
    if not {"L1 docs", "L2 docs", "L1+L2 docs"}.issubset(reg):
        continue
    gain = reg["L1+L2 docs"] - max(reg["L1 docs"], reg["L2 docs"])
    if "en" in langs_by_pair[pair]:
        en.append(gain)
    else:
        non_en.append(gain)

def summarize_gains(vals):
    mean = sum(vals) / len(vals)
    gt0 = sum(1 for v in vals if v > 0)
    gt01 = sum(1 for v in vals if v > 0.1)
    return mean, gt0, gt01, len(vals)

print_section("Input coverage")
print(f"Pairs with full regimes (L1, L2, L1+L2): {len(valid_pairs)}")
print(f"Pairs missing regimes (excluded): {len(missing_pairs)}")
if missing_pairs:
    print("Missing pairs:", ", ".join(missing_pairs))

print_section("Global picture: delta distribution (nDCG@10)")
deltas = [row["delta_ndcg"] for row in global_rows if not math.isnan(row["delta_ndcg"])]
pos = sum(1 for v in deltas if v > 0)
neg = sum(1 for v in deltas if v < 0)
zero = sum(1 for v in deltas if abs(v) < 1e-12)
n = len(deltas)
mean_delta = mean(deltas)
median_delta = median(deltas)
min_delta = min(deltas) if deltas else float("nan")
max_delta = max(deltas) if deltas else float("nan")
print(f"Groups (pair, doc setting): {n}")
print(f"Delta>0: {pos}/{n} ({(pos / n * 100):.1f}%), Delta<0: {neg}/{n} ({(neg / n * 100):.1f}%), Delta=0: {zero}/{n}")
print(f"Mean delta: {mean_delta:.4f} (norm={mean_delta / 100:.4f})")
print(f"Median delta: {median_delta:.4f} (norm={median_delta / 100:.4f})")
print(f"Range: {min_delta:.4f} to {max_delta:.4f} (norm={min_delta / 100:.4f} to {max_delta / 100:.4f})")
if deltas:
    max_row = max(global_rows, key=lambda r: r["delta_ndcg"])
    min_row = min(global_rows, key=lambda r: r["delta_ndcg"])
    max_lambda = max_row["lambda_star_mid"]
    min_lambda = min_row["lambda_star_mid"]
    max_lambda_fmt = f"{max_lambda:.0f} ({max_lambda / 100:.2f})" if not math.isnan(max_lambda) else "nan"
    min_lambda_fmt = f"{min_lambda:.0f} ({min_lambda / 100:.2f})" if not math.isnan(min_lambda) else "nan"
    print(f"Max gain: pair={max_row['pair']}, docs={max_row['doc_mix']}, delta={max_row['delta_ndcg']:.4f}, lambda*={max_lambda_fmt}")
    print(f"Most negative: pair={min_row['pair']}, docs={min_row['doc_mix']}, delta={min_row['delta_ndcg']:.4f}, lambda*={min_lambda_fmt}")

print_section("Finding 1: English in index split")
en_present = [r for r in global_rows if r["en_in_index"]]
en_absent = [r for r in global_rows if not r["en_in_index"]]
def split_summary(rows):
    vals = [r["delta_ndcg"] for r in rows if not math.isnan(r["delta_ndcg"])]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    return mean(vals), min(vals), max(vals), len(vals)
en_mean, en_min, en_max, en_n = split_summary(en_present)
abs_mean, abs_min, abs_max, abs_n = split_summary(en_absent)
print(f"EN present: n={en_n}, mean={en_mean:.4f} (norm={en_mean / 100:.4f}), min={en_min:.4f}, max={en_max:.4f}")
print(f"EN absent: n={abs_n}, mean={abs_mean:.4f} (norm={abs_mean / 100:.4f}), min={abs_min:.4f}, max={abs_max:.4f}")
print(f"EN absent all delta>0: {all(r['delta_ndcg'] > 0 for r in en_absent)}")

print_section("Finding 2: English as strongest partner (monolingual docs)")
mono_rows = [r for r in global_rows if r["doc_type"] == "mono" and r["doc_lang"]]
doc_partner = defaultdict(dict)
for r in mono_rows:
    doc_lang = r["doc_lang"]
    partner = r["lang_b"] if doc_lang == r["lang_a"] else r["lang_a"]
    doc_partner[doc_lang][partner] = r["delta_ndcg"]
doc_langs = sorted([dl for dl in doc_partner.keys() if dl != "en"])
en_best = 0
en_total = 0
for dl in doc_langs:
    partners = doc_partner[dl]
    if "en" not in partners:
        continue
    en_total += 1
    en_delta = partners["en"]
    non_en_partners = sorted(
        [(p, v) for p, v in partners.items() if p != "en"],
        key=lambda x: x[1],
        reverse=True,
    )
    best_non = non_en_partners[0] if non_en_partners else ("-", float("nan"))
    second_non = non_en_partners[1] if len(non_en_partners) > 1 else None
    best_partner = max(partners.items(), key=lambda x: x[1])
    if best_partner[0] == "en":
        en_best += 1
    second_str = ""
    if second_non:
        second_str = f", second_nonEN={second_non[0].upper()} {second_non[1]:.4f}"
    print(f"{dl.upper()} docs: EN {en_delta:.4f}, best_nonEN={best_non[0].upper()} {best_non[1]:.4f}{second_str}, EN_is_best={best_partner[0] == 'en'}")
print(f"EN best among partners: {en_best}/{en_total}")

print_section("Finding 4: Bilingual indexing gains (best_mixed_ndcg)")
print("Gain = best_mixed_ndcg(L1+L2 docs) - max(best_mixed_ndcg(L1 docs), best_mixed_ndcg(L2 docs))")
print("Scale: 0-100 nDCG points (divide by 100 to match 0-1 scale)")
non_mean, non_gt0, non_gt01, non_n = summarize_gains(non_en)
en_mean, en_gt0, en_gt01, en_n = summarize_gains(en)
print(f"Non-EN pairs (n={non_n}): mean={non_mean:.4f}, gains>0={non_gt0}, gains>0.1={non_gt01}")
print(f"EN pairs (n={en_n}): mean={en_mean:.4f}, gains>0={en_gt0}, gains>0.1={en_gt01}")

# --- Metric deltas across (pair, doc_mix) from pivot ---
pivot_groups = defaultdict(list)
with open(pivot_path, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["pair"] in valid_pairs:
            if row.get("method") and row["method"] != "embed":
                continue
            pivot_groups[(row["pair"], row["doc_mix"])].append(row)

metrics = ["ndcg10", "mrr10", "r10"]
deltas_by_metric = {m: [] for m in metrics}
delta_by_setting = defaultdict(dict)
overall_best = {}

for (pair, doc_mix), rows in pivot_groups.items():
    endpoints = [r for r in rows if is_endpoint(r["mix_ratio"])]
    midpoints = [r for r in rows if not is_endpoint(r["mix_ratio"])]
    best_val = None
    best_ratio = None
    for r in rows:
        val = to_float(r["ndcg10"])
        ratio = to_float(r["mix_ratio"])
        if math.isnan(val) or math.isnan(ratio):
            continue
        if best_val is None or val > best_val or (val == best_val and ratio < best_ratio):
            best_val = val
            best_ratio = ratio
    overall_best[(pair, doc_mix)] = {"best_ndcg10": best_val, "lambda_star_all": best_ratio}

    for m in metrics:
        best_endpoint = max((to_float(r[m]) for r in endpoints), default=float("nan"))
        if not midpoints:
            delta = 0.0
        else:
            best_mixed = max(to_float(r[m]) for r in midpoints)
            delta = best_mixed - (best_endpoint if not math.isnan(best_endpoint) else 0.0)
        deltas_by_metric[m].append(delta)
        delta_by_setting[(pair, doc_mix)][m] = delta

def summarize_deltas(vals):
    mean = sum(vals) / len(vals)
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    zero = sum(1 for v in vals if abs(v) < 1e-12)
    return mean, pos, neg, zero, len(vals)

metric_labels = {
    "ndcg10": "nDCG@10",
    "mrr10": "MRR@10",
    "r10": "Recall@10",
}

print_section("Finding 3: Monolingual endpoint alignment (nDCG@10)")
match_better = 0
match_worse = 0
match_tie = 0
missing = 0
violations = []
for row in mono_rows:
    key = (row["pair"], row["doc_mix"])
    rows = pivot_groups.get(key, [])
    if not rows:
        missing += 1
        continue
    endpoints = {}
    for r in rows:
        ratio = to_float(r["mix_ratio"])
        if is_endpoint(ratio):
            endpoints[ratio] = to_float(r["ndcg10"])
    if 0.0 not in endpoints or 100.0 not in endpoints:
        missing += 1
        continue
    if row["doc_regime"] == "L1 docs":
        match_val = endpoints[0.0]
        opp_val = endpoints[100.0]
    elif row["doc_regime"] == "L2 docs":
        match_val = endpoints[100.0]
        opp_val = endpoints[0.0]
    else:
        missing += 1
        continue
    if match_val > opp_val:
        match_better += 1
    elif match_val < opp_val:
        match_worse += 1
        violations.append((row["pair"], row["doc_mix"], match_val, opp_val))
    else:
        match_tie += 1
print(f"Monolingual settings: {len(mono_rows)}")
print(f"Match endpoint better: {match_better}, worse: {match_worse}, ties: {match_tie}, missing: {missing}")
if violations:
    print("Violations (pair, docs, match, opposite):")
    for pair, doc_mix, mv, ov in violations[:10]:
        print(f"  {pair}, {doc_mix}: match={mv:.4f}, opposite={ov:.4f}")

print_section("Finding 3: Peak location counts (nDCG@10)")
def count_peaks(rows, use_p_doc):
    counts = defaultdict(int)
    missing_local = 0
    for row in rows:
        key = (row["pair"], row["doc_mix"])
        best = overall_best.get(key)
        if not best or best["lambda_star_all"] is None or math.isnan(best["lambda_star_all"]):
            missing_local += 1
            continue
        lam_ratio = best["lambda_star_all"] / 100.0
        if use_p_doc:
            if row["doc_regime"] == "L1 docs":
                p_doc = 1.0 - lam_ratio
            elif row["doc_regime"] == "L2 docs":
                p_doc = lam_ratio
            else:
                missing_local += 1
                continue
            key_val = round(p_doc, 1)
        else:
            key_val = round(lam_ratio, 1)
        counts[key_val] += 1
    return counts, missing_local

non_en_mono = [
    r for r in global_rows
    if r["doc_type"] == "mono" and r["lang_a"] != "en" and r["lang_b"] != "en"
]
en_pair_en_docs = [
    r for r in global_rows
    if r["doc_type"] == "mono" and r["doc_lang"] == "en"
]
en_pair_non_en_docs = [
    r for r in global_rows
    if r["doc_type"] == "mono" and r["doc_lang"] != "en" and "en" in (r["lang_a"], r["lang_b"])
]
non_en_bi = [
    r for r in global_rows
    if r["doc_type"] == "bi" and r["lang_a"] != "en" and r["lang_b"] != "en"
]
en_bi = [
    r for r in global_rows
    if r["doc_type"] == "bi" and "en" in (r["lang_a"], r["lang_b"])
]

def print_counts(label, rows, use_p_doc):
    counts, missing_local = count_peaks(rows, use_p_doc)
    items = ", ".join(f"{k:.1f}: {counts[k]}" for k in sorted(counts.keys()))
    print(f"{label}: n={len(rows)}, missing={missing_local}, counts={{ {items} }}")

print_counts("Non-EN pairs, monolingual index (p_doc)", non_en_mono, True)
print_counts("EN pairs, EN-only index (p_doc)", en_pair_en_docs, True)
print_counts("EN pairs, non-EN-only index (p_doc)", en_pair_non_en_docs, True)
print_counts("Non-EN pairs, bilingual index (lambda)", non_en_bi, False)
print_counts("EN pairs, bilingual index (lambda)", en_bi, False)

print_section("Mixing effects across metrics (delta = best interior - best endpoint)")
print("Scale: 0-100 metric points (divide by 100 to match 0-1 scale)")
for m in metrics:
    mean_val, pos, neg, zero, n = summarize_deltas(deltas_by_metric[m])
    label = metric_labels.get(m, m)
    print(f"{label}: mean={mean_val:.4f} (norm={mean_val/100:.4f}), +={pos}, -={neg}, =0 {zero}, n={n}")

print_section("EN-pair nuance: delta nDCG<0 but delta R@10>0")
nuance = []
violations = []
for key, deltas in delta_by_setting.items():
    nd = deltas.get("ndcg10")
    rr = deltas.get("r10")
    if nd is None or rr is None:
        continue
    if nd < 0 and rr > 0:
        nuance.append(key)
        pair, doc_mix = key
        info = setting_info.get(key)
        is_en_pair = "en" in pair_langs.get(pair, ("", ""))
        en_in_index = info["en_in_index"] if info else False
        if not (is_en_pair and en_in_index):
            violations.append(key)
print(f"Settings with ndcg<0 & r10>0: {len(nuance)}")
print(f"Subset check (EN pairs + EN in index): violations={len(violations)}")
if violations:
    print("Violations:")
    for pair, doc_mix in violations[:10]:
        print(f"  {pair}, {doc_mix}")

print_section("Headroom effect (Spearman rho: best endpoint vs delta)")
def spearman_summary(rows, label):
    x = [r["best_endpoint_ndcg"] for r in rows]
    y = [r["delta_ndcg"] for r in rows]
    if len(x) < 2:
        print(f"{label}: n={len(x)}, rho=nan")
        return
    rho = spearman_rho(x, y)
    print(f"{label}: n={len(x)}, rho={rho:.3f}")

en_pair_rows = [r for r in global_rows if "en" in (r["lang_a"], r["lang_b"])]
non_en_index_rows = [r for r in global_rows if not r["en_in_index"]]
spearman_summary(global_rows, "All settings")
spearman_summary(en_pair_rows, "EN pairs")
spearman_summary(non_en_index_rows, "EN absent in index")

# --- Language factor probes on controlled subset ---
controlled = [
    row for row in processed_rows
    if row["doc_type"] == "mono"
    and row["lang_a"] != "en"
    and row["lang_b"] != "en"
]

print_section("Language factor probes (controlled subset)")
print("Subset: non-EN/non-EN pairs + monolingual docs")
print(f"Settings={len(controlled)}")
print("Stats treat each (pair, doc_lang) setting as one sample")
print("Scale: 0-100 nDCG points (divide by 100 to match 0-1 scale)")

if controlled:
    controlled_by_pair = defaultdict(list)
    for r in controlled:
        controlled_by_pair[r["pair"]].append(r)

    def stat_spearman(records, x_key):
        x = [r[x_key] for r in records]
        y = [r["delta_ndcg"] for r in records]
        return spearman_rho(x, y)

    # Typology: lang2vec_knn
    rho, lo, hi, n_ok = cluster_bootstrap_stats(
        controlled_by_pair,
        lambda recs: stat_spearman(recs, "lang2vec_knn"),
    )
    print(
        "Typology (lang2vec_knn): "
        f"rho={rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
        f"n_settings={len(controlled)}, n_pairs={len(controlled_by_pair)}, n_boot={n_ok}"
    )

    # Family proxy: glot_tree
    rho, lo, hi, n_ok = cluster_bootstrap_stats(
        controlled_by_pair,
        lambda recs: stat_spearman(recs, "glot_tree"),
    )
    print(
        "Family (glot_tree): "
        f"rho={rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
        f"n_settings={len(controlled)}, n_pairs={len(controlled_by_pair)}, n_boot={n_ok}"
    )

    # Script match vs mismatch
    script_groups = defaultdict(list)
    for r in controlled:
        script_groups[r["script_match"]].append(r["delta_ndcg"])
    if len(script_groups) == 2:
        def stat_script_diff(records):
            match = [r["delta_ndcg"] for r in records if r["script_match"] == "match"]
            mismatch = [r["delta_ndcg"] for r in records if r["script_match"] == "mismatch"]
            if not match or not mismatch:
                return float("nan")
            return mean(match) - mean(mismatch)

        diff, lo, hi, n_ok = cluster_bootstrap_stats(controlled_by_pair, stat_script_diff)
        for k in sorted(script_groups.keys()):
            print(f"Script {k}: n={len(script_groups[k])}, mean={fmt_mean(mean(script_groups[k]))}")
        print(
            "Script mean diff (match - mismatch): "
            f"{fmt_mean(diff)}, 95% CI [{lo:.4f}, {hi:.4f}], n_boot={n_ok}"
        )
    else:
        print("Script: not enough groups for a match/mismatch comparison")

    # Resource pattern (multi-group)
    resource_groups = defaultdict(list)
    for r in controlled:
        resource_groups[r["resource_pattern"]].append(r["delta_ndcg"])
    for k in sorted(resource_groups.keys()):
        print(f"Resource {k}: n={len(resource_groups[k])}, mean={fmt_mean(mean(resource_groups[k]))}")
    # Exploratory association measures (no p-values): eta^2 / omega^2 with cluster-bootstrap CIs.
    def stat_resource_eta(records):
        groups = defaultdict(list)
        for r in records:
            groups[r["resource_pattern"]].append(r["delta_ndcg"])
        return eta_squared(groups)

    def stat_resource_omega(records):
        groups = defaultdict(list)
        for r in records:
            groups[r["resource_pattern"]].append(r["delta_ndcg"])
        return omega_squared(groups)

    e2, lo, hi, n_ok = cluster_bootstrap_stats(controlled_by_pair, stat_resource_eta)
    print(f"Resource association eta^2: {e2:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], n_boot={n_ok}")
    w2, lo, hi, n_ok = cluster_bootstrap_stats(controlled_by_pair, stat_resource_omega)
    print(f"Resource association omega^2: {w2:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], n_boot={n_ok}")

    # Optional: treat resource pattern as an ordinal "number of high-resource languages" (H-H=2, mixed=1, L-L=0).
    resource_index = {"H-H": 2.0, "H-L": 1.0, "L-H": 1.0, "L-L": 0.0}
    def stat_resource_index_rho(records):
        x = [resource_index.get(r["resource_pattern"], float("nan")) for r in records]
        y = [r["delta_ndcg"] for r in records]
        pairs = [(a, b) for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b)]
        if len(pairs) < 2:
            return float("nan")
        x2, y2 = zip(*pairs)
        return spearman_rho(list(x2), list(y2))

    rho, lo, hi, n_ok = cluster_bootstrap_stats(controlled_by_pair, stat_resource_index_rho)
    print(f"Resource index (0/1/2) Spearman rho: {rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], n_boot={n_ok}")

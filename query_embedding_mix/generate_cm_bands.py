#!/usr/bin/env python
"""
generate_cm_bands.py — Code-mix by bands with caching & resume.

Key features
  • Bands (connected 0–100)
  • One API call per query returns ONLY missing bands (resumable)
  • Fluency-first prompt with explicit K per band; smart retries (inclusive edges)
  • Opportunistic filing: write ONLY truly mixed outputs; never force-fill
  • Optional concurrency (--workers)
  • qids-common.tsv (intersection of qids across ALL bands for this run)
  • --cache_dir to reuse previously generated (qid, band) from older runs
         e.g., cache_dir can point at your old dev.full outputs
  • NEW: --qid_list TSV with 'qid' (or 'id') column to filter which ids to process
  • NEW: Token counting uses mix_count.py (count_two_langs) for source/target and CM outputs.

Install:
  pip install --upgrade openai pandas tqdm python-dotenv tenacity
  # and for mix_count.py dependencies:
  pip install stanza regex langid
  # download the stanza tokenizers you need for your source/target languages
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from openai import OpenAI
from mix_count import count_two_langs

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ───────────────────────── config ─────────────────────────

# Pricing (USD / 1M tokens) — purely indicative
PRICE_USD_PER_M_TOKEN = {
    "gpt-5":       {"in": 1.25, "out": 10.00},
    "gpt-5-mini":  {"in": 0.25, "out":  2.00},
    "gpt-5-nano":  {"in": 0.05, "out":  0.40},
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out":  0.60},
}
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
ZERO_PRICE = {"in": 0.0, "out": 0.0}

DEFAULT_MAX_TRIES = 2
TEMP_FIRST = 0.0             # used only if the model supports it
TEMP_RETRY = 0.0              # used only if the model supports it

# ───────────────────────── regex ─────────────────────────

# HAN_RUN_RE = re.compile(r"[\u4E00-\u9FFF]+")                        # Han runs
# EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")                # English word

LOG_LEVELS = {"warn": 1, "info": 2, "debug": 3}

# Models with fixed sampling (omit temperature/top_p)
FIXED_SAMPLING_REGEX = re.compile(r"^(gpt-5|o1|o3)", re.IGNORECASE)

# Globals initialized in main
band_files: Dict[Tuple[int, int], Any] = {}
band_locks: Dict[Tuple[int, int], threading.Lock] = {}

# Track (qid, band) written in this run + any preloaded (resume/cache)
written_set: set[Tuple[str, Tuple[int,int]]] = set()
written_lock = threading.Lock()

# Per-band qid sets (for qids-common.tsv)
band_qids: Dict[Tuple[int,int], set[str]] = {}
band_qids_lock = threading.Lock()

cost_lock = threading.Lock()
total_cost = 0.0

# ───────────────────────── token counting via mix_count ─────────────────────────

def counter_base_lang(src_lang: str, tgt_lang: str) -> str:
    if "zh" in (src_lang, tgt_lang):
        return "zh"
    if "hi" in (src_lang, tgt_lang):
        return "hi"
    return "en"

def src_token_count(text: str, src_lang: str, tgt_lang: str) -> int:
    counts = count_two_langs(
        text,
        tgt_lang,
        src_lang,
        drop_digit_tokens=True,
        base_lang=counter_base_lang(src_lang, tgt_lang),
    )
    return int(counts.get(src_lang, 0))

def tgt_share_ratio(s: str, src_lang: str, tgt_lang: str) -> float:
    """Target-language share (%) based on mix_count.count_two_langs."""
    counts = count_two_langs(
        s,
        tgt_lang,
        src_lang,
        drop_digit_tokens=True,
        base_lang=counter_base_lang(src_lang, tgt_lang),
    )
    tgt = int(counts.get(tgt_lang, 0))
    src = int(counts.get(src_lang, 0))
    denom = tgt + src
    return (100.0 * tgt / denom) if denom else 0.0

def build_user_message(src_label: str, tgt_label: str, src_text: str, tgt_text: str) -> str:
    return f'{src_label}: "{src_text}"\n{tgt_label}: "{tgt_text}"'

# ───────────────────────── helpers ─────────────────────────

def parse_bands(bands: List[str]) -> List[Tuple[int, int]]:
    """Parse 'L-H' strings, ensure 0..100 coverage and exact connectivity."""
    out: List[Tuple[int, int]] = []
    for b in bands:
        if "-" not in b:
            sys.exit(f"Band '{b}' must be like 'L-H'")
        L, H = b.split("-", 1)
        try:
            L, H = int(L), int(H)
        except ValueError:
            sys.exit(f"Band '{b}' must be integers like '40-70'")
        if not (0 <= L < H <= 100):
            sys.exit(f"Band '{b}' must satisfy 0 <= L < H <= 100")
        out.append((L, H))
    out_sorted = sorted(out, key=lambda x: x[0])
    if out_sorted[0][0] != 0 or out_sorted[-1][1] != 100:
        sys.exit("Bands must cover [0,100] (first L==0 and last H==100).")
    for (L1, H1), (L2, H2) in zip(out_sorted, out_sorted[1:]):
        if L2 != H1:
            sys.exit(f"Bands must connect exactly (gap/overlap: {L1}-{H1} then {L2}-{H2}).")
    return out_sorted

def relpath(path, base):
    path, base = pathlib.Path(path), pathlib.Path(base)
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)

def find_band_for_ratio(r: float, bands: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Map a measured ratio to the band it belongs to (bands cover [0,100])."""
    for L, H in bands:
        if L <= r <= H:
            return (L, H)
    return bands[0] if r < bands[0][0] else bands[-1]

def midpoint(band: Tuple[int, int]) -> float:
    L, H = band
    return int(round((L + H) / 2.0))


def initial_K_for_band(L: int, H: int, src_word_count: int) -> int:
    """Initial K from band midpoint, clamped to [1, src-1] for true mix."""
    if src_word_count <= 1:
        return 1
    target_pct = midpoint((L, H))
    k = round(target_pct / 100.0 * src_word_count)
    return max(1, min(k, src_word_count - 1))

# inclusive boundaries + minimum push of 1 token
EPS = 1e-6
def adjust_K(current_K: int, measured_ratio: float, L: int, H: int, src_word_count: int) -> int:
    """Move K toward band; always change by at least 1 when possible."""
    if src_word_count <= 1:
        return current_K
    if measured_ratio <= L + EPS:
        delta_pct = max(0.0, L - measured_ratio)
        delta_K = max(1, ceil(delta_pct / 100.0 * src_word_count))
        newK = current_K + delta_K
    elif measured_ratio >= H - EPS:
        delta_pct = max(0.0, measured_ratio - H)
        delta_K = max(1, ceil(delta_pct / 100.0 * src_word_count))
        newK = current_K - delta_K
    else:
        return current_K
    lower, upper = 1, max(1, src_word_count - 1)
    return min(max(newK, lower), upper)

EXAMPLE_BY_PAIR: Dict[str, str] = {
    "EN-ZH": (
        "Given:\n"
        '  EN: "What are the causes of volcanic eruptions?"\n'
        '  ZH: "火山噴發的原因有哪些?"\n'
        "Output:\n"
        '{\n'
        '  "0-20": "What are the 原因 of volcanic eruptions?",\n'
        '  "20-40": "What are the 原因 of 火山 eruptions?",\n'
        '  "40-60": "What are the 原因 of 火山噴發?",\n'
        '  "60-80": "What are 火山噴發的原因?",\n'
        '  "80-100": "火山噴發的原因有 what?"\n'
        "}\n"
    ),
    "EN-VI": (
        "Given:\n"
        '  EN: "What are the causes of volcanic eruptions?"\n'
        '  VI: "Nguyên nhân gây ra các vụ phun trào núi lửa là gì?"\n'
        "Output:\n"
        '{\n'
        '  "0-20": "What are the nguyên nhân of volcanic eruptions?",\n'
        '  "20-40": "What are the nguyên nhân of núi lửa eruptions?",\n'
        '  "40-60": "What are the nguyên nhân của phun trào núi lửa?",\n'
        '  "60-80": "Nguyên nhân gây ra volcanic eruptions là gì?",\n'
        '  "80-100": "Nguyên nhân của phun trào núi lửa là what?"\n'
        "}\n"
    ),
    "ZH-VI": (
        "Given:\n"
        '  ZH: "火山噴發的原因有哪些?"\n'
        '  VI: "Nguyên nhân gây ra các vụ phun trào núi lửa là gì?"\n'
        "Output:\n"
        '{\n'
        '  "0-20": "火山噴發的 nguyên nhân有哪些?",\n'
        '  "20-40": "火山 phun trào的 nguyên nhân有哪些?",\n'
        '  "40-60": "火山 phun trào的 nguyên nhân là gì?",\n'
        '  "60-80": "Nguyên nhân gây ra 火山 phun trào là gì?",\n'
        '  "80-100": "Nguyên nhân gây ra các vụ phun trào núi lửa有哪些?"\n'
        "}\n"
    ),
    "ID-VI": (
        "Given:\n"
        '  ID: "Apa penyebab letusan gunung berapi?"\n'
        '  VI: "Nguyên nhân gây ra các vụ phun trào núi lửa là gì?"\n'
        "Output:\n"
        '{\n'
        '  "0-20": "Apa penyebab letusan núi lửa?",\n'
        '  "20-40": "Apa penyebab phun trào gunung berapi?",\n'
        '  "40-60": "Apa nguyên nhân phun trào núi lửa?",\n'
        '  "60-80": "Nguyên nhân gây ra letusan gunung berapi là gì?",\n'
        '  "80-100": "Nguyên nhân gây ra các vụ phun trào gunung berapi là gì?"\n'
        "}\n"
    ),
    "HI-ID": (
        "Given:\n"
        '  HI: "ज्वालामुखी विस्फोट के कारण क्या हैं?"\n'
        '  ID: "Apa penyebab letusan gunung berapi?"\n'
        "Output:\n"
        '{\n'
        '  "0-20": "ज्वालामुखी विस्फोट के penyebab क्या हैं?",\n'
        '  "20-40": "ज्वालामुखी letusan gunung berapi के कारण क्या हैं?",\n'
        '  "40-60": "ज्वालामुखी letusan के penyebab क्या हैं?",\n'
        '  "60-80": "Apa penyebab ज्वालामुखी विस्फोट के?",\n'
        '  "80-100": "Apa penyebab letusan gunung berapi के कारण?"\n'
        "}\n"
    ),
    "EN-AR": (
        "Given:\n"
        '  EN: "What are the causes of volcanic eruptions?"\n'
        '  AR: "ما أسباب ثوران البراكين؟"\n'
        "Output:\n"
        '{\n'
        '  "0-20": "What are the أسباب of volcanic eruptions?",\n'
        '  "20-40": "What are the أسباب of ثوران البراكين?",\n'
        '  "40-60": "ما أسباب ثوران volcanoes؟",\n'
        '  "60-80": "ما هي أسباب volcanic eruptions؟",\n'
        '  "80-100": "أسباب ثوران البراكين هي what؟"\n'
        "}\n"
    ),
}


def get_pair_example(src_label: str, tgt_label: str) -> str:
    return EXAMPLE_BY_PAIR.get(f"{src_label}-{tgt_label}", "")

def build_system_prompt_all_bands(
    bands: List[Tuple[int,int]], K_map: Dict[str, int], src_word_count: int,
    src_label: str, tgt_label: str
) -> str:
    """
    Fluency-first multi-band instruction with explicit K per band.
    Ask for controlled source-language retention while prioritizing fluency.
    """
    band_labels = [f"{L}-{H}" for (L, H) in bands]
    k_spec_lines = [f'  • "{lab}": use exactly {src_word_count - K_map[lab]} {src_label} words' for lab in band_labels]
    k_spec = "\n".join(k_spec_lines)
    example = get_pair_example(src_label, tgt_label)
    example_block = f"For example:\n{example}" if example else ""

    return (
        "You are a bilingual re-writer.\n"
        "Return a JSON object where each key is a band label and each value is ONE fluent, natural "
        f"code-mixed sentence derived ONLY from the given {src_label} & {tgt_label} pair "
        "(reuse words/phrases; do not invent facts).\n"
        "Code-mixing is the intra-sentence blending of two or more languages—injecting words, morphemes, or grammar from one language into an utterance in another. The generated sentence should not be just a concatenation of two original sentences; you should not repeat words of the same meaning from different languages\n"
        f"Bands to produce: {', '.join(band_labels)}.\n"
        "Fluency and Accuracy are the top priority. Preserve the original meaning fully with all information present. Avoid choppy, word-by-word alternation.\n"
        # "Prefer replacing words for content words/phrases (nouns/verbs/adjectives); avoid function words.\n"
        "Ensure the Code-mixing is smooth and seamless, with good grammar and syntax in both languages.\n"
        f"You should consider reordering or replacing a {src_label} word with its {tgt_label} counterpart "
        "(and vice-versa) to achieve best fluency.\n"
        "Target constraints per band:\n"
        f"{k_spec}\n"
        f"Keep overall length roughly similar to the original sentence; "
        "small deviations are fine if more natural.\n"
        "Strictly output JSON only with exactly these keys and string values. No extra commentary.\n"
        f"{example_block}"
    )


def relpath(path, base):
    path, base = pathlib.Path(path), pathlib.Path(base)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return os.path.relpath(path, base)


# ───────── JSON/response helpers ─────────

def _best_effort_extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try strict json.loads; if it fails, extract the first {...} span and parse."""
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = t[l:r+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


# ───────── OpenAI wrappers (robust; handle model caps) ─────────

def _extract_text_from_response(resp: Any) -> str:
    """Pull text from Responses API object."""
    try:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass
    data: Any = None
    try:
        data = resp.model_dump()
    except Exception:
        data = getattr(resp, "__dict__", None)

    def walk(x):
        outs = []
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("text", "output_text") and isinstance(v, str):
                    outs.append(v)
                else:
                    outs.extend(walk(v))
        elif isinstance(x, list):
            for y in x:
                outs.extend(walk(y))
        return outs

    if data is not None:
        texts = [t for t in walk(data) if isinstance(t, str) and t.strip()]
        if texts:
            return "\n".join(texts)
    return ""


def _extract_usage(resp: Any) -> Tuple[int, int]:
    """(prompt_tokens, completion_tokens) if present; else (0,0)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    try:
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        return pt, ct
    except Exception:
        pass
    try:
        d = usage.model_dump()
    except Exception:
        d = getattr(usage, "__dict__", {}) or {}
    return int(d.get("prompt_tokens", 0) or 0), int(d.get("completion_tokens", 0) or 0)


def _extract_chat_content(message_content: Any) -> str:
    """Handle both plain-string and list-of-parts chat responses."""
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def model_basename(model: str) -> str:
    """Accept provider/model ids such as openai/gpt-5-mini."""
    return model.strip().rsplit("/", 1)[-1]


def resolve_price_cfg(model: str) -> Dict[str, float]:
    return PRICE_USD_PER_M_TOKEN.get(model) or PRICE_USD_PER_M_TOKEN.get(model_basename(model)) or ZERO_PRICE


def _supports_sampling(model: str) -> bool:
    """False for fixed sampling models (e.g., GPT-5 / o1 / o3)."""
    return FIXED_SAMPLING_REGEX.match(model_basename(model)) is None


def _sampling_kwargs(model: str, temp_for_attempt: float) -> Dict[str, Any]:
    """Include sampling kwargs only when supported."""
    if _supports_sampling(model):
        return {"temperature": temp_for_attempt, "top_p": 0.8}
    return {}


@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def call_openai_responses(
    *, client: OpenAI, model: str, instructions: str,
    user_text: str, temp: float
):
    sampling = _sampling_kwargs(model, temp)
    base_kwargs = dict(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        **sampling,
    )
    # Prefer JSON mode; if rejected, retry without it.
    try:
        return client.responses.create(response_format={"type": "json_object"}, **base_kwargs)
    except Exception as e:
        msg = str(e)
        if "response_format" in msg or "unsupported_parameter" in msg:
            return client.responses.create(**base_kwargs)
        raise


@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def call_openai_chat_fallback(
    *, client: OpenAI, model: str, system: str,
    user_text: str, temp: float,
    extra_headers: Optional[Dict[str, str]] = None,
):
    sampling = _sampling_kwargs(model, temp)
    base_kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_text}],
        **sampling,
    )
    if extra_headers:
        base_kwargs["extra_headers"] = extra_headers
    # Try JSON mode; if rejected, retry plain.
    try:
        return client.chat.completions.create(response_format={"type": "json_object"}, **base_kwargs)
    except Exception as e:
        msg = str(e)
        if "response_format" in msg or "unsupported_parameter" in msg:
            return client.chat.completions.create(**base_kwargs)
        raise


def token_cost(prompt_toks: int, completion_toks: int, price: dict) -> float:
    return (prompt_toks * price["in"] + completion_toks * price["out"]) / 1_000_000


# ───────────────────────── core processing ─────────────────────────

def process_one_query(qid: str, src_text: str, tgt_text: str,
                      all_bands: List[Tuple[int,int]],
                      pending_bands: List[Tuple[int,int]],
                      src_lang: str, tgt_lang: str,
                      src_label: str, tgt_label: str,
                      model: str, temp_first: float, temp_retry: float,
                      price_cfg: dict, log_level: int, client: OpenAI,
                      extra_headers: Optional[Dict[str, str]],
                      max_tries: int, fsync: bool):
    """Generate ONLY missing bands for one source/target pair with at most max_tries calls."""
    global total_cost

    def log(level: str, msg: str):
        if LOG_LEVELS[level] <= log_level:
            tqdm.tqdm.write(msg)

    if not pending_bands:
        return

    src_word_count = src_token_count(src_text, src_lang, tgt_lang)
    user_msg = build_user_message(src_label, tgt_label, src_text, tgt_text)

    if src_word_count == 0:
        log("warn", f"[{qid}] {src_label} has 0 words; skipping (no outputs written).")
        return

    # Track pending & per-band K
    pending = list(pending_bands)
    K_map: Dict[Tuple[int,int], int] = {
        band: initial_K_for_band(band[0], band[1], src_word_count) for band in pending_bands
    }

    for attempt in range(1, max_tries + 1):
        if not pending:
            break  # all requested bands already got a valid mixed result (opportunistic filing)

        # Prompt only for remaining bands
        K_for_labels = {f"{L}-{H}": K_map[(L, H)] for (L, H) in pending}
        sys_prompt = build_system_prompt_all_bands(
            pending, K_for_labels, src_word_count, src_label, tgt_label
        )
        temp = (temp_first if attempt == 1 else temp_retry)

        # OpenRouter currently works most reliably through Chat Completions.
        chat = call_openai_chat_fallback(
            client=client, model=model, system=sys_prompt,
            user_text=user_msg, temp=temp, extra_headers=extra_headers
        )
        try:
            raw = _extract_chat_content(chat.choices[0].message.content)
        except Exception:
            raw = ""
        try:
            u = chat.usage
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            pt, ct = 0, 0

        with cost_lock:
            total_cost += token_cost(pt, ct, price_cfg)

        obj = _best_effort_extract_json_object(raw or "")
        if not isinstance(obj, dict):
            obj = {}

        still_pending: List[Tuple[int,int]] = []
        for band in pending:
            L, H = band
            key = f"{L}-{H}"
            txt = (obj.get(key) or "").strip()

            # if len(txt) < 5:
            #     # too short; keep pending (no writing)
            #     log("debug", f"[{qid} {key} K={K_map[band]}] attempt {attempt}: empty/too short → retry")
            #     still_pending.append(band)
            #     continue

            r = tgt_share_ratio(txt, src_lang, tgt_lang)

            # Opportunistic filing: write only if truly mixed (0<r<100) and not already present
            if 0.0 < r < 100.0:
                actual_band = find_band_for_ratio(r, all_bands)
                with written_lock:
                    missing_here = (qid, actual_band) not in written_set
                    if missing_here:
                        written_set.add((qid, actual_band))
                if missing_here:
                    with band_locks[actual_band]:
                        band_files[actual_band].write(f"{qid}\t{txt}\n")
                        band_files[actual_band].flush()
                        if fsync:
                            os.fsync(band_files[actual_band].fileno())
                        with band_qids_lock:
                            band_qids[actual_band].add(qid)
                    log("info", f"[{qid}] filed: ratio={r:.1f}% → {actual_band[0]}-{actual_band[1]}")

            # If in target band and truly mixed, this target is satisfied.
            if (0.0 < r < 100.0) and (L <= r <= H):
                continue  # already written under its actual band (equals target band here)

            # Otherwise adjust K and keep pending (if more attempts left)
            if attempt < max_tries:
                newK = adjust_K(K_map[band], r, L, H, src_word_count)
                dirc = "↑" if newK > K_map[band] else ("↓" if newK < K_map[band] else "=")
                log("debug", f"[{qid} {key}] attempt {attempt}: ratio={r:.1f}% → adjust K {dirc} {K_map[band]}→{newK}")
                K_map[band] = newK
                still_pending.append(band)
            else:
                # Final attempt: per rule, DO NOT write if not truly mixed or out-of-band.
                log("warn", (f"[{qid} {key} K={K_map[band]}] max tries reached; "
                             f"final ratio={r:.1f}% → no write (not mixed/out-of-band)"))

        pending = still_pending


# ───────────────────────── resume + cache support ─────────────────────────

def _open_output_files(out_dir: pathlib.Path, bands: List[Tuple[int,int]], buffering: int = 1):
    """Open/append band files; initialize locks & band_qids; preload existing qids for resume."""
    for band in bands:
        p = out_dir / f"queries-cm{band[0]}-{band[1]}.tsv"
        band_qids[band] = set()
        band_locks[band] = threading.Lock()
        if p.exists():
            # Parse existing qids (resume)
            try:
                with p.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        parts = line.split("\t", 1)
                        if not parts:
                            continue
                        qid = parts[0]
                        if qid:
                            band_qids[band].add(qid)
                            written_set.add((qid, band))
            except Exception:
                pass
            band_files[band] = p.open("a", encoding="utf-8", buffering=buffering)
        else:
            band_files[band] = p.open("a", encoding="utf-8", buffering=buffering)


def _load_cache(cache_dir: pathlib.Path, bands: List[Tuple[int,int]], log_level: int) -> Dict[Tuple[int,int], Dict[str, str]]:
    """Load cached (qid -> text) per band from a previous run directory."""
    cache: Dict[Tuple[int,int], Dict[str, str]] = {}
    for band in bands:
        cache[band] = {}
        p = cache_dir / f"queries-cm{band[0]}-{band[1]}.tsv"
        if not p.exists():
            if LOG_LEVELS["info"] <= log_level:
                tqdm.tqdm.write(f"[i] cache missing: {p}")
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if not parts:
                    continue
                qid = parts[0]
                txt = parts[1] if len(parts) > 1 else ""
                if qid:
                    cache[band][qid] = txt
    return cache


def _prefill_from_cache(ids: List[str],
                        cache: Dict[Tuple[int,int], Dict[str, str]],
                        bands: List[Tuple[int,int]],
                        fsync: bool,
                        log_level: int):
    """Copy cached lines for current ids into THIS run's output files (skip if already present)."""
    copied = 0
    for qid in ids:
        for band in bands:
            if (qid, band) in written_set:
                continue  # already present in current out_dir
            cached_txt = cache.get(band, {}).get(qid)
            if not cached_txt:
                continue
            # Write cached line to current outputs
            with written_lock:
                written_set.add((qid, band))
            with band_locks[band]:
                band_files[band].write(f"{qid}\t{cached_txt}\n")
                band_files[band].flush()
                if fsync:
                    os.fsync(band_files[band].fileno())
                with band_qids_lock:
                    band_qids[band].add(qid)
            copied += 1
    if LOG_LEVELS["info"] <= log_level:
        tqdm.tqdm.write(f"[i] cache prefill: copied {copied} (qid,band) lines into this run.")


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--src", "--en", dest="src", required=True, type=pathlib.Path,
                    help="Source language query TSV (old alias: --en)")
    ap.add_argument("--tgt", "--zh", dest="tgt", required=True, type=pathlib.Path,
                    help="Target language query TSV (old alias: --zh)")
    ap.add_argument("--src_lang", default="en",
                    help="ISO language code for source TSV (e.g., en)")
    ap.add_argument("--tgt_lang", default="zh",
                    help="ISO language code for target TSV (e.g., zh, vi, ar)")
    ap.add_argument("--out_dir", default="queries_cm_bands", type=pathlib.Path)
    ap.add_argument("--bands", nargs="+", default=["0-40", "40-70", "70-100"],
                    help="Connected bands covering [0,100], e.g., 0-40 40-70 70-100")
    ap.add_argument("--model", default="openai/gpt-5-mini")
    ap.add_argument("--temperature", type=float, default=TEMP_FIRST,
                    help="Temperature for generation if the model supports it")
    ap.add_argument("--env_file", type=pathlib.Path, default=PROJECT_ROOT / ".env",
                    help="Optional .env file containing OPENROUTER_API_KEY / OPENAI_API_KEY")
    ap.add_argument("--api_base", default=DEFAULT_API_BASE,
                    help="API base URL. Defaults to OpenRouter.")
    ap.add_argument("--http_referer", default=os.getenv("OPENROUTER_HTTP_REFERER", ""),
                    help="Optional HTTP-Referer header for OpenRouter attribution")
    ap.add_argument("--x_title", default=os.getenv("OPENROUTER_X_TITLE", ""),
                    help="Optional X-Title header for OpenRouter attribution")
    ap.add_argument("--max_rows", type=int, help="Process only first N rows")
    ap.add_argument("--log", choices=list(LOG_LEVELS.keys()), default="warn",
                    help="Logging verbosity")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (1 = no parallelism)")
    ap.add_argument("--max_tries", type=int, default=DEFAULT_MAX_TRIES,
                    help="Attempts per query (only for bands still pending)")
    ap.add_argument("--cache_dir", type=pathlib.Path,
                    help="Directory of a previous run to reuse (qid, band) outputs from (e.g., dev.full)")
    ap.add_argument("--fsync", action="store_true",
                    help="fsync on each write for durability (slower)")
    ap.add_argument("--qid_list", type=pathlib.Path,
                    help="Optional TSV with a 'qid' (or 'id') column; only process these IDs")

    args = ap.parse_args()

    log_level = LOG_LEVELS[args.log]
    price_cfg = resolve_price_cfg(args.model)
    max_tries = max(1, int(args.max_tries))
    args.src_lang = args.src_lang.strip().lower()
    args.tgt_lang = args.tgt_lang.strip().lower()
    if not args.src_lang or not args.tgt_lang:
        sys.exit("Both --src_lang and --tgt_lang are required.")
    if args.src_lang == args.tgt_lang:
        sys.exit("--src_lang and --tgt_lang must be different.")
    src_label = args.src_lang.upper()
    tgt_label = args.tgt_lang.upper()

    bands = parse_bands(args.bands)

    # IO
    a = pd.read_csv(args.src, sep="\t", names=["id", "src"], dtype=str)
    b = pd.read_csv(args.tgt, sep="\t", names=["id", "tgt"], dtype=str)
    df = a.merge(b, on="id", how="inner").sort_values("id")
    if df.empty:
        sys.exit("No overlapping IDs between the two TSV files.")

    # Optional: filter by qid list TSV
    if args.qid_list:
        qid_df = pd.read_csv(args.qid_list, sep="\t", dtype=str)
        qid_col = None
        for cand in ["qid", "id", "QID", "Id", "ID"]:
            if cand in qid_df.columns:
                qid_col = cand
                break
        if qid_col is None:
            sys.exit("qid_list TSV must contain a 'qid' (or 'id') column.")
        keep = set(qid_df[qid_col].astype(str).tolist())
        before = len(df)
        df = df[df["id"].astype(str).isin(keep)].copy()
        after = len(df)
        if LOG_LEVELS["info"] <= log_level:
            tqdm.tqdm.write(f"[i] qid_list filter: kept {after} of {before} overlapping IDs.")
        if df.empty:
            sys.exit("After applying qid_list filter, no IDs remain.")

    if args.max_rows:
        df = df.head(args.max_rows)

    ids, srcs, tgts = df["id"].tolist(), df["src"].tolist(), df["tgt"].tolist()

    # outputs (resume-aware)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _open_output_files(args.out_dir, bands, buffering=1)

    # Optional cache preload (e.g., from dev.full)
    if args.cache_dir:
        cache = _load_cache(args.cache_dir, bands, log_level)
        _prefill_from_cache(ids, cache, bands, fsync=args.fsync, log_level=log_level)
    else:
        if LOG_LEVELS["info"] <= log_level:
            tqdm.tqdm.write("[i] no cache_dir provided; generating everything that’s missing.")

    # Init OpenRouter / OpenAI-compatible client
    load_dotenv(args.env_file)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY).")
    extra_headers: Dict[str, str] = {}
    if args.http_referer:
        extra_headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        extra_headers["X-Title"] = args.x_title
    client = OpenAI(base_url=args.api_base, api_key=api_key)
    if price_cfg == ZERO_PRICE and LOG_LEVELS["info"] <= log_level:
        tqdm.tqdm.write(f"[i] no pricing configured for model '{args.model}'; cost estimate will remain $0.0000.")

    # Build per-qid pending band lists (only generate what's missing now)
    per_qid_pending: Dict[str, List[Tuple[int,int]]] = {}
    for qid in ids:
        missing = [band for band in bands if (qid, band) not in written_set]
        if missing:
            per_qid_pending[qid] = missing

    total_to_process = len(per_qid_pending)
    already_done = len(ids) - total_to_process
    if already_done:
        print(f"[i] Ready: {already_done} qids fully satisfied by resume/cache; "
              f"{total_to_process} qids still have missing bands.")

    # Parallel or sequential processing
    with tqdm.tqdm(total=total_to_process, desc="CM by bands (resume+cache)", unit="qry") as bar:
        if args.workers <= 1:
            for qid, src_text, tgt_text in zip(ids, srcs, tgts):
                pending_bands = per_qid_pending.get(qid)
                if not pending_bands:
                    continue
                process_one_query(qid, src_text, tgt_text, bands, pending_bands,
                                  args.src_lang, args.tgt_lang, src_label, tgt_label,
                                  args.model, args.temperature, TEMP_RETRY,
                                  price_cfg, log_level, client, extra_headers or None,
                                  max_tries=max_tries, fsync=args.fsync)
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = []
                for qid, src_text, tgt_text in zip(ids, srcs, tgts):
                    pending_bands = per_qid_pending.get(qid)
                    if not pending_bands:
                        continue
                    futs.append(ex.submit(process_one_query, qid, src_text, tgt_text, bands, pending_bands,
                                          args.src_lang, args.tgt_lang, src_label, tgt_label,
                                          args.model, args.temperature, TEMP_RETRY,
                                          price_cfg, log_level, client, extra_headers or None,
                                          max_tries, args.fsync))
                for _ in as_completed(futs):
                    bar.update(1)

    # Close band files
    for band, f in band_files.items():
        f.close()
        print(f"[✓] wrote {relpath((args.out_dir / f'queries-cm{band[0]}-{band[1]}.tsv'), pathlib.Path.cwd())}")

    # Write the intersection of qids across all bands (based on what’s written now)
    band_to_qids: Dict[Tuple[int,int], set[str]] = {}
    for band in bands:
        path = args.out_dir / f"queries-cm{band[0]}-{band[1]}.tsv"
        s: set[str] = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    qid = line.split("\t", 1)[0]
                    if qid:
                        s.add(qid)
        band_to_qids[band] = s

    common_qids: Optional[set[str]] = None
    for band in bands:
        s = band_to_qids[band]
        common_qids = set(s) if common_qids is None else (common_qids & s)
    common_qids = common_qids or set()

    common_path = args.out_dir / "qids-common.tsv"
    with common_path.open("w", encoding="utf-8") as fo:
        for q in sorted(common_qids, key=lambda x: (len(x), x)):
            fo.write(f"{q}\n")
    print(f"[✓] wrote {relpath(common_path, pathlib.Path.cwd())} with {len(common_qids)} qids in all bands")

    global total_cost
    print(f"Done — estimated total cost ≈ ${total_cost:,.4f}")

if __name__ == "__main__":
    main()

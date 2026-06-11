# mix_count.py
# pip install stanza regex langid
# (once per language you use: python -c "import stanza; stanza.download('en'); stanza.download('zh'); stanza.download('fr')")

from collections import Counter
import re
import regex as ure
import stanza
from langid.langid import LanguageIdentifier, model

# --- caches so you can call this function many times efficiently ---
_PIPELINES = {}         # base_lang -> stanza.Pipeline
_LID_CACHE = {}         # (lang1, lang2) sorted tuple -> LanguageIdentifier

# --- Unicode-aware regexes ---
RE_SPLIT = ure.compile(r'(\p{Script=Han}+|\p{Script=Latin}+|[^\p{Script=Han}\p{Script=Latin}\s]+)')
RE_HAS_LETTER = ure.compile(r'\p{L}+')
RE_URL   = re.compile(r'https?://\S+|www\.\S+', re.I)
RE_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
RE_HANDLE= re.compile(r'[@#]\w+')

def _get_pipeline(base_lang: str):
    """Lazy-load a UD tokenizer for the chosen base language."""
    if base_lang not in _PIPELINES:
        _PIPELINES[base_lang] = stanza.Pipeline(
            base_lang, processors="tokenize", tokenize_pretokenized=False, verbose=False
        )
    return _PIPELINES[base_lang]

def _get_identifier(lang1: str, lang2: str):
    """Two-language langid identifier (thread-unsafe global cache by design)."""
    key = tuple(sorted((lang1, lang2)))
    if key not in _LID_CACHE:
        ident = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        ident.set_languages(list(key))
        _LID_CACHE[key] = ident
    return _LID_CACHE[key]

def _script_split(token: str):
    """
    Split a token into contiguous script runs so no piece mixes Han and Latin.
    (For same-script pairs like EN–FR this usually leaves tokens unchanged.)
    """
    return [m.group(0) for m in RE_SPLIT.finditer(token) if m.group(0).strip()]

def _is_language_word(tok: str, drop_digit_tokens: bool) -> bool:
    """Keep only word-like tokens; drop URLs/emails/handles; optionally drop anything with digits."""
    if not RE_HAS_LETTER.search(tok):
        return False
    if RE_URL.search(tok) or RE_EMAIL.search(tok) or RE_HANDLE.search(tok):
        return False
    if drop_digit_tokens and any(ch.isdigit() for ch in tok):
        return False
    return True

def count_two_langs(
    text: str,
    lang1: str,
    lang2: str,
    *,
    drop_digit_tokens: bool = True,
    base_lang: str | None = None,
    return_tokens: bool = False,
):
    """
    Count word tokens for exactly TWO languages in a mixed sentence.

    Parameters
    ----------
    text : str
        Input text (single sentence or short query).
    lang1, lang2 : str
        ISO 639-1 codes for the two target languages (e.g., 'en', 'zh', 'fr').
        Only tokens classified as one of these two are counted.
    drop_digit_tokens : bool, default True
        If True, drop tokens that contain any digits (e.g., '3pm', 'A380').
    base_lang : str or None, default None
        Stanza UD tokenizer language to use as the base splitter.
        If None, uses 'zh' if either language is 'zh', else 'en'.
    return_tokens : bool, default False
        If True, also return a list of (token, language) pairs that were counted.

    Returns
    -------
    counts : dict
        {lang1: int, lang2: int}
    tokens : list[tuple[str, str]]  (only if return_tokens=True)
        The kept tokens paired with their predicted language.
    """
    if base_lang is None:
        base_lang = "zh" if ("zh" in (lang1, lang2)) else "en"

    nlp = _get_pipeline(base_lang)
    ident = _get_identifier(lang1, lang2)

    # 1) UD tokenization
    doc = nlp(text)
    raw_tokens = [tok.text for sent in doc.sentences for tok in sent.tokens]

    # 2) Enforce script-boundary splits (prevents mixed-script tokens like 'Walmart买')
    pieces = []
    for tok in raw_tokens:
        pieces.extend(_script_split(tok))

    # 3) Filter & 4) per-token LID among the two languages
    counts = Counter({lang1: 0, lang2: 0})
    kept = []
    for t in pieces:
        if not _is_language_word(t, drop_digit_tokens):
            continue
        lab, _ = ident.classify(t)
        if lab == lang1 or lab == lang2:
            counts[lab] += 1
            if return_tokens:
                kept.append((t, lab))

    return (counts, kept) if return_tokens else counts


# Example usage
if __name__ == "__main__":
#     texts = """Where is 格拉芬堡点 located?
# where did hip hop/rap come from
# which bbc 广播 电台 specializes in sports commentaries?
# 哪些 B 维生素 help with serotonin levels?
# Where is 桃金娘's house?
# Which 轴 in a chart displays the 标签 for the 数据点?
# which 修正案 can u insist on a 陪审团 trial
# where was 沃尔玛 founded?
# Where is 显卡 位置 in the CPU?
# Where was Robert Anderson 出生 and when?
# Which amendment of the US Constitution led to women gaining the 选举权?""".split("\n")
    texts = [
        "where did hip hop/rap come from",
        "Where did 嘻哈 /说唱 come from",
        "嘻哈/说唱从何而来"
    ]

    for txt in texts:
        counts = count_two_langs(txt, "zh", "en", drop_digit_tokens=True)
        total = counts["zh"] + counts["en"]
        mix_percent = 0.0 if total == 0 else 100.0 * counts["en"] / total  # L2 into L1 example
        print(counts, mix_percent)




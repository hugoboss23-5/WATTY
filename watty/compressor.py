"""
Watty Semantic Compressor
=========================
Compresses memory text to a denser representation while preserving
meaning for AI consumption. Embeddings stay untouched — recall
quality is identical. The text just gets smaller and sharper.

Compression pipeline:
  1. Normalize — collapse whitespace, fix encoding artifacts
  2. Strip filler — remove words that add no information
  3. Phrase shorthand — common phrases → compact notation
  4. Dedup sentences — remove repeated information within a chunk
  5. Code detection — skip compression for code blocks (already dense)
  6. Notation — for fact-like memories, convert to key:value format

Design principle: compressed text must still be readable by an AI.
This is semantic compression, not gzip.

Hugo & Rim · February 2026
"""

import re


# ── Filler words (zero information density) ──────────────
# These are safe to remove without changing meaning.
# Kept conservative — only words that truly add nothing.
FILLER_WORDS = frozenset({
    # Articles (low-info for AI consumption, safe to strip)
    'the', 'a', 'an',
    # Hedging adverbs (zero information density)
    'very', 'really', 'just', 'also', 'quite', 'rather',
    'somewhat', 'actually', 'basically', 'essentially', 'practically',
    # Certainty adverbs (emphasis without information)
    'certainly', 'definitely', 'obviously', 'clearly', 'simply',
    'literally', 'virtually', 'generally', 'typically', 'usually',
    # NOTE: Removed — these carry meaning and were corrupting memories:
    #   Verbs: is, are, was, were, be, been, being, have, has, had, do, does, did
    #   Pronouns: it, its, itself
    #   Spatial: there, here
})

# Words to NEVER strip (even if they look like filler in other contexts)
KEEP_WORDS = frozenset({
    'not', 'no', 'never', 'none', 'nor', 'neither',
    'but', 'however', 'although', 'though', 'yet',
    'if', 'then', 'else', 'when', 'while', 'until',
    'all', 'each', 'every', 'both', 'either',
    'must', 'should', 'always',
})


# ── Phrase → shorthand mappings ──────────────────────────
SHORTHANDS = [
    (r'\bfor example\b', 'e.g.'),
    (r'\bthat is to say\b', 'i.e.'),
    (r'\bin order to\b', 'to'),
    (r'\bas well as\b', '+'),
    (r'\bin addition to\b', '+'),
    (r'\bon the other hand\b', 'alternatively'),
    (r'\bat the same time\b', 'simultaneously'),
    (r'\bwith respect to\b', 're:'),
    (r'\bin terms of\b', 're:'),
    (r'\bdue to the fact that\b', 'because'),
    (r'\bfor the purpose of\b', 'for'),
    (r'\bin the event that\b', 'if'),
    (r'\bat this point in time\b', 'now'),
    (r'\bprior to\b', 'before'),
    (r'\bsubsequent to\b', 'after'),
    (r'\bin spite of\b', 'despite'),
    (r'\bwith regard to\b', 're:'),
    (r'\ba large number of\b', 'many'),
    (r'\ba small number of\b', 'few'),
    (r'\bthe majority of\b', 'most'),
    (r'\bthe fact that\b', 'that'),
    (r'\bit is important to note that\b', 'note:'),
    (r'\bit should be noted that\b', 'note:'),
    (r'\bplease note that\b', 'note:'),
    (r'\bas a result of\b', 'from'),
    (r'\bin the context of\b', 'in'),
    (r'\bwith the exception of\b', 'except'),
    (r'\bfor the most part\b', 'mostly'),
    (r'\btake into account\b', 'consider'),
    (r'\bmake use of\b', 'use'),
    (r'\bis able to\b', 'can'),
    (r'\bis not able to\b', "can't"),
    (r'\bin the process of\b', 'during'),
]

# Precompile for speed
_SHORTHAND_PATTERNS = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in SHORTHANDS]


# ── Code detection ───────────────────────────────────────
CODE_INDICATORS = [
    r'^\s*(def |class |import |from |if |for |while |return |async |await )',
    r'^\s*(function |const |let |var |module\.exports|require\()',
    r'^\s*[{}\[\]();]',
    r'^\s*#\s*(include|define|ifdef|pragma)',
    r'^\s*<[a-zA-Z/]',  # HTML/XML tags
    r'[=!<>]{2,}',  # Operators
    r'\b(self\.|this\.|console\.|print\(|printf\()',
]
_CODE_PATTERNS = [re.compile(p, re.MULTILINE) for p in CODE_INDICATORS]


def _is_code(text: str) -> bool:
    """Detect if text is primarily code (skip compression)."""
    # Quick heuristics
    lines = text.strip().split('\n')
    if not lines:
        return False

    code_lines = 0
    for line in lines[:20]:  # Check first 20 lines
        for pat in _CODE_PATTERNS:
            if pat.search(line):
                code_lines += 1
                break

    return code_lines / max(len(lines[:20]), 1) > 0.4


def _is_structured_data(text: str) -> bool:
    """Detect JSON, YAML, TOML, config files."""
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        return True
    if stripped.startswith('[') and stripped.endswith(']'):
        return True
    # TOML/YAML headers
    if re.match(r'^\[[\w.-]+\]', stripped):
        return True
    return False


# ── Compression Pipeline ─────────────────────────────────

def compress(content: str, aggressive: bool = False) -> tuple[str, float]:
    """
    Compress memory text semantically.

    Returns (compressed_text, compression_ratio).
    Ratio < 1.0 means text got smaller.
    Ratio of 1.0 means no compression (code/structured data).

    aggressive=True applies heavier compression for old memories.
    """
    if not content or len(content) < 20:
        return content, 1.0

    # Skip code and structured data — already dense
    if _is_code(content) or _is_structured_data(content):
        # Just normalize whitespace for code
        result = _normalize(content)
        return result, len(result) / len(content)

    original_len = len(content)

    # Pass 1: Normalize
    text = _normalize(content)

    # Pass 2: Strip filler words
    text = _strip_filler(text, aggressive=aggressive)

    # Pass 3: Apply phrase shorthands
    text = _apply_shorthands(text)

    # Pass 4: Deduplicate sentences
    text = _dedup_sentences(text)

    # Pass 5: Collapse redundant whitespace (final cleanup)
    text = re.sub(r'  +', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    ratio = len(text) / max(original_len, 1)
    return text, round(ratio, 3)


def _normalize(text: str) -> str:
    """Normalize whitespace, fix encoding artifacts."""
    # Fix common encoding artifacts
    text = text.replace('\u00e2\u0080\u0099', "'")
    text = text.replace('\u00e2\u0080\u0093', '-')
    text = text.replace('\u00e2\u0080\u0094', '-')
    text = text.replace('\u00e2\u0080\u009c', '"')
    text = text.replace('\u00e2\u0080\u009d', '"')
    text = text.replace('\u00ef\u00bb\u00bf', '')  # BOM
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')

    # Collapse runs of spaces (but preserve newlines for structure)
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _strip_filler(text: str, aggressive: bool = False) -> str:
    """Remove filler words that carry no information."""
    words = text.split()

    # Don't strip from very short texts
    if len(words) < 8:
        return text

    # In aggressive mode, also strip articles and demonstratives
    extra_fillers = frozenset({'this', 'that', 'these', 'those', 'which', 'who', 'whom'}) if aggressive else frozenset()
    all_fillers = FILLER_WORDS | extra_fillers

    result = []
    for i, word in enumerate(words):
        lower = word.lower().rstrip('.,;:!?')

        # Never strip keep-words
        if lower in KEEP_WORDS:
            result.append(word)
            continue

        # Don't strip if it's the start of a sentence (capitalized after period)
        if i > 0 and words[i-1].endswith('.') and word[0].isupper():
            result.append(word)
            continue

        # Strip if it's a filler
        if lower in all_fillers:
            continue

        result.append(word)

    return ' '.join(result)


def _apply_shorthands(text: str) -> str:
    """Replace verbose phrases with compact equivalents."""
    for pattern, replacement in _SHORTHAND_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _dedup_sentences(text: str) -> str:
    """Remove duplicate or near-duplicate sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text

    seen = set()
    unique = []

    for s in sentences:
        # Normalize for comparison
        key = re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()
        # Skip empty
        if not key:
            continue

        # Check for near-duplicates (first 50 chars)
        short_key = key[:50]
        if short_key in seen:
            continue

        seen.add(short_key)
        unique.append(s)

    return ' '.join(unique)


# ── Batch compression for dream cycle ────────────────────

def compress_batch(memories: list[dict], aggressive: bool = False) -> list[dict]:
    """
    Compress a batch of memories.

    Input: list of {'id': int, 'content': str, 'memory_tier': str}
    Output: list of {'id': int, 'compressed': str, 'ratio': float, 'savings_chars': int}
    """
    results = []
    for mem in memories:
        content = mem.get('content', '')
        compressed, ratio = compress(content, aggressive=aggressive)
        savings = len(content) - len(compressed)

        results.append({
            'id': mem['id'],
            'original_len': len(content),
            'compressed': compressed,
            'compressed_len': len(compressed),
            'ratio': ratio,
            'savings_chars': savings,
        })

    return results


def estimate_savings(memories: list[dict]) -> dict:
    """Dry run: estimate how much compression would save without modifying anything."""
    total_original = 0
    total_compressed = 0
    code_skipped = 0
    compressible = 0

    for mem in memories:
        content = mem.get('content', '')
        total_original += len(content)

        if _is_code(content) or _is_structured_data(content):
            total_compressed += len(content)
            code_skipped += 1
        else:
            compressed, ratio = compress(content)
            total_compressed += len(compressed)
            compressible += 1

    savings = total_original - total_compressed
    ratio = total_compressed / max(total_original, 1)

    return {
        'total_memories': len(memories),
        'compressible': compressible,
        'code_skipped': code_skipped,
        'original_size_chars': total_original,
        'compressed_size_chars': total_compressed,
        'savings_chars': savings,
        'savings_pct': round((1 - ratio) * 100, 1),
        'overall_ratio': round(ratio, 3),
    }

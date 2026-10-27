"""Text helpers: normalization, tokenization, n-grams, Jaccard similarity.

Everything here is deterministic and dependency free. Tokens are
lowercase word characters; numbers are kept as tokens.
"""

import re

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text):
    """Lowercase, collapse whitespace, strip surrounding space."""
    return " ".join(str(text).lower().split())


def tokens(text):
    """Word tokens of the normalized text."""
    return _WORD.findall(normalize(text))


def ngrams(text, n=5):
    """Set of word n-grams (as tuples) of the normalized text."""
    toks = tokens(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(set_a, set_b):
    """Jaccard similarity of two sets. 0.0 when either set is empty."""
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    if shared == 0:
        return 0.0
    return shared / len(set_a | set_b)


def template_key(prompt, prefix_words=4):
    """Template key for a prompt: its first N normalized words, with
    pure numbers replaced by '#'.

    Four words is the right length for the job: prompt templates put
    the slot value late ("Write a product description for ...",
    "Summarize this article about ..."), so the first four words are
    the template and the slot stays out. Prompts that start with the
    topic ("What is photosynthesis?") keep their own key, which is
    correct: they are not a repeated template.
    """
    toks = tokens(prompt)[:prefix_words]
    # Mask pure numbers ("article 12" and "article 99" are the same
    # template). Tokens that merely contain a digit, like "p1" or
    # "gpt4", are identifiers, not numbers, and are left alone.
    masked = ["#" if tok.isdigit() else tok for tok in toks]
    return " ".join(masked)

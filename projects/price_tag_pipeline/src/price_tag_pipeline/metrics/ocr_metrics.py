"""OCR-quality metrics: CER (character error rate), WER (word error rate).

Pure-Python Levenshtein — no external dependency. For large eval sets prefer
`jiwer` (faster), but we want this to work in any environment.
"""

from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate. Returns 0.0 if both empty, 1.0 if reference empty and hyp not."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(hypothesis, reference) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    # Word-level Levenshtein with sequences instead of characters.
    a, b = hyp_words, ref_words
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cost = 0 if wa == wb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1] / len(ref_words)


def mean_cer(hypotheses: list[str], references: list[str]) -> float:
    assert len(hypotheses) == len(references)
    if not references:
        return 0.0
    return sum(cer(h, r) for h, r in zip(hypotheses, references)) / len(references)


def mean_wer(hypotheses: list[str], references: list[str]) -> float:
    assert len(hypotheses) == len(references)
    if not references:
        return 0.0
    return sum(wer(h, r) for h, r in zip(hypotheses, references)) / len(references)

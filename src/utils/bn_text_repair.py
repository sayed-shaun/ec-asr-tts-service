"""Repairs a wav2vec2 CTC decoding artifact: Bengali dependent vowel signs
(matras) and virama decoded as their own space-separated "word" instead of
attached to the preceding consonant ("কাছ েেে কিছুই" instead of "কাছে কিছুই").

These characters are grammatically required to attach directly to a base
consonant with zero space, so any run of them preceded by whitespace is
unambiguously a decoding artifact, never legitimate text — safe to repair
unconditionally rather than needing an allowlist like itn.py's numerals.
"""

import re

_VOWEL_SIGNS = "া-ৌ্"
_REPEATED_SIGN_RE = re.compile(rf"([{_VOWEL_SIGNS}])\1+")
_STRAY_SIGN_RE = re.compile(rf"\s+([{_VOWEL_SIGNS}]+)")


def repair_stray_vowel_signs(text: str) -> str:
    """Collapse repeated vowel-sign runs, then reattach stray ones to the
    preceding word by removing the whitespace before them.
    """
    if not text:
        return text
    text = _REPEATED_SIGN_RE.sub(r"\1", text)
    text = _STRAY_SIGN_RE.sub(r"\1", text)
    return text

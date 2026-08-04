"""Inverse text normalization for Bengali numerals: spelled-out numbers in
the model's output rewritten as digits.

The checkpoint always spells numbers out ("আটশো দুই"), while real Bengali
transcripts write them as digits ("802") — measured on this project's FLEURS
benchmark, 220 of 242 digit-bearing references use ASCII digits and none of
1322 hypotheses contain a digit at all. That formatting gap accounts for 38%
of all word errors despite digit-bearing utterances being only 18% of the
corpus.

Design is deliberately conservative: NUMERALS is an allowlist, and any token
not in it terminates the run and passes through byte-identical. A missing or
misspelled numeral word therefore causes a missed conversion, never corrupted
text — the failure mode is "did nothing", not "mangled a sentence".
"""

import re

# Bengali 0-99 is largely irregular, so it needs a table rather than
# composition rules. Spelling variants observed in this model's output are
# listed alongside the standard form.
_SMALL = {
    "শূন্য": 0, "এক": 1, "দুই": 2, "দুি": 2, "তিন": 3, "চার": 4,
    "পাঁচ": 5, "পাচ": 5, "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9,
    "দশ": 10, "এগারো": 11, "এগার": 11, "বারো": 12, "বার": 12,
    "তেরো": 13, "তের": 13, "চৌদ্দ": 14, "পনেরো": 15, "পনের": 15,
    "ষোলো": 16, "ষোল": 16, "সতেরো": 17, "সতের": 17, "আঠারো": 18,
    "আঠার": 18, "উনিশ": 19, "ঊনিশ": 19,
    "বিশ": 20, "একুশ": 21, "বাইশ": 22, "তেইশ": 23, "চব্বিশ": 24,
    "পঁচিশ": 25, "ছাব্বিশ": 26, "সাতাশ": 27, "আঠাশ": 28,
    "উনত্রিশ": 29, "ঊনত্রিশ": 29,
    "ত্রিশ": 30, "একত্রিশ": 31, "বত্রিশ": 32, "তেত্রিশ": 33,
    "চৌত্রিশ": 34, "পঁয়ত্রিশ": 35, "ছত্রিশ": 36, "সাঁইত্রিশ": 37,
    "আটত্রিশ": 38, "উনচল্লিশ": 39, "ঊনচল্লিশ": 39,
    "চল্লিশ": 40, "একচল্লিশ": 41, "বিয়াল্লিশ": 42, "তেতাল্লিশ": 43,
    "চুয়াল্লিশ": 44, "পঁয়তাল্লিশ": 45, "ছেচল্লিশ": 46,
    "সাতচল্লিশ": 47, "আটচল্লিশ": 48, "উনপঞ্চাশ": 49, "ঊনপঞ্চাশ": 49,
    "পঞ্চাশ": 50, "একান্ন": 51, "বায়ান্ন": 52, "তিপ্পান্ন": 53,
    "চুয়ান্ন": 54, "পঞ্চান্ন": 55, "ছাপ্পান্ন": 56, "সাতান্ন": 57,
    "আটান্ন": 58, "উনষাট": 59, "ঊনষাট": 59,
    "ষাট": 60, "একষট্টি": 61, "বাষট্টি": 62, "তেষট্টি": 63,
    "চৌষট্টি": 64, "পঁয়ষট্টি": 65, "ছেষট্টি": 66, "সাতষট্টি": 67,
    "আটষট্টি": 68, "উনসত্তর": 69, "ঊনসত্তর": 69,
    "সত্তর": 70, "একাত্তর": 71, "বাহাত্তর": 72, "তিয়াত্তর": 73,
    "চুয়াত্তর": 74, "পঁচাত্তর": 75, "ছিয়াত্তর": 76, "সাতাত্তর": 77,
    "আটাত্তর": 78, "উনআশি": 79, "ঊনআশি": 79,
    "আশি": 80, "একাশি": 81, "বিরাশি": 82, "তিরাশি": 83, "চুরাশি": 84,
    "পঁচাশি": 85, "ছিয়াশি": 86, "সাতাশি": 87, "আটাশি": 88,
    "উননব্বই": 89, "ঊননব্বই": 89,
    "নব্বই": 90, "একানব্বই": 91, "বিরানব্বই": 92, "তিরানব্বই": 93,
    "চুরানব্বই": 94, "পঁচানব্বই": 95, "ছিয়ানব্বই": 96,
    "সাতানব্বই": 97, "আটানব্বই": 98, "নিরানব্বই": 99,
}

# Ordered longest-first so "শত" is tried before "শ" when splitting compounds.
_SCALES = {
    "কোটি": 10_000_000,
    "লক্ষ": 100_000,
    "লাখ": 100_000,
    "হাজার": 1_000,
    "শত": 100,
    "শো": 100,
    "শ": 100,
}

_DECIMAL = {"দশমিক"}

_NUMERAL_TOKENS = set(_SMALL) | set(_SCALES) | _DECIMAL

# Bengali digits appear in some references; normalize them for parsing only.
_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _expand(token: str) -> list[str] | None:
    """Split a compound numeral like "আটশো" into ["আট", "শো"].

    The model writes hundreds as one word, so without this the whole token
    misses the allowlist and the number is never converted. Returns None for
    anything that is not a numeral, which is what keeps non-numeric text
    untouched.
    """
    if token in _NUMERAL_TOKENS:
        return [token]
    for scale, _ in sorted(_SCALES.items(), key=lambda kv: -len(kv[0])):
        if token.endswith(scale) and len(token) > len(scale):
            head = token[: -len(scale)]
            if head in _SMALL:
                return [head, scale]
    return None


def _parse_int(tokens: list[str]) -> int | None:
    """Accumulate numeral tokens into one integer, or None if unparseable."""
    total = current = 0
    seen = False
    for token in tokens:
        small = _SMALL.get(token)
        if small is not None:
            current += small
            seen = True
            continue
        scale = _SCALES.get(token)
        if scale is None:
            return None
        # "শো" multiplies what precedes it and stays in the running value;
        # হাজার/লাখ/কোটি close off a group and commit it to the total.
        if scale == 100:
            current = (current or 1) * 100
        else:
            total += (current or 1) * scale
            current = 0
        seen = True
    return total + current if seen else None


def _render_run(tokens: list[str], min_value: int) -> str | None:
    """Render one numeral run as digits, or None to leave it spelled out."""
    if _DECIMAL & set(tokens):
        idx = next(i for i, t in enumerate(tokens) if t in _DECIMAL)
        whole_tokens, frac_tokens = tokens[:idx], tokens[idx + 1 :]
        if not frac_tokens:
            return None
        whole = _parse_int(whole_tokens) if whole_tokens else 0
        if whole is None:
            return None
        # "দুই দশমিক চার" -> 2.4; digits after the point are read out in
        # sequence rather than as one quantity.
        frac_parts = []
        for token in frac_tokens:
            value = _parse_int([token])
            if value is None:
                return None
            frac_parts.append(str(value))
        return f"{whole}.{''.join(frac_parts)}"

    value = _parse_int(tokens)
    if value is None:
        return None
    has_scale = any(t in _SCALES for t in tokens)
    # A bare small numeral is usually prose ("এক ব্যক্তি"), where references
    # keep the word; only convert it once it is large enough or carries a
    # scale word to read as a real quantity.
    if not has_scale and value < min_value:
        return None
    return str(value)


def bengali_numerals_to_digits(text: str, min_value: int = 10) -> str:
    """Rewrite spelled-out Bengali numbers in `text` as ASCII digits.

    min_value gates bare numerals with no scale word: below it they are left
    as words, since small numbers in running prose are normally spelled out.
    Set 0 to convert every numeral, or a very large value to convert only
    those carrying শো/হাজার/লাখ/কোটি.
    """
    if not text:
        return text

    parts = re.split(r"(\s+)", text)
    out = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if not part.strip():
            out.append(part)
            i += 1
            continue

        # Collect the longest run of consecutive numeral words.
        run_tokens: list[str] = []
        run_end = i
        j = i
        while j < len(parts):
            if not parts[j].strip():
                j += 1
                continue
            expanded = _expand(parts[j].translate(_BENGALI_DIGITS))
            if expanded is None:
                break
            run_tokens.extend(expanded)
            run_end = j
            j += 1

        if not run_tokens:
            out.append(part)
            i += 1
            continue

        rendered = _render_run(run_tokens, min_value)
        if rendered is None:
            out.extend(parts[i : run_end + 1])
        else:
            out.append(rendered)
        i = run_end + 1

    return "".join(out)

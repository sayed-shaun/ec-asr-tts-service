"""Split arbitrary text into bounded, speakable clauses.

indic-parler-tts is autoregressive over a single prompt: generation time and
prosody both degrade on long inputs, and it silently truncates past its token
budget. IncrementalTextChunker keeps the streaming interface it was ported
with, so a future streaming endpoint can reuse it; chunk_text() is the
one-shot wrapper /synthesize uses.
"""

STRONG_BOUNDARIES = frozenset("।॥.!?\n")
SOFT_BOUNDARIES = frozenset(",;:，؛")


class IncrementalTextChunker:
    """Turn arbitrary text deltas into bounded, speakable clauses."""

    def __init__(self, max_chars: int = 160, min_soft_chars: int = 32) -> None:
        if max_chars < 16:
            raise ValueError("max_chars must be at least 16")
        self.max_chars = max_chars
        self.min_soft_chars = min(min_soft_chars, max_chars)
        self._buffer = ""

    @property
    def pending(self) -> str:
        return self._buffer

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        return self._extract()

    def flush(self) -> list[str]:
        chunks = self._extract(force=True)
        self._buffer = ""
        return chunks

    def clear(self) -> None:
        self._buffer = ""

    def _extract(self, force: bool = False) -> list[str]:
        chunks: list[str] = []
        while self._buffer:
            boundary = self._find_boundary()
            if boundary is not None:
                self._emit(chunks, boundary + 1)
                continue
            if len(self._buffer) >= self.max_chars:
                split_at = self._hard_split()
                self._emit(chunks, split_at)
                continue
            break

        if force and self._buffer.strip():
            chunks.append(self._buffer.strip())
            self._buffer = ""
        return chunks

    def _find_boundary(self) -> int | None:
        """Index of the clause boundary to cut at, never past max_chars."""
        for index, char in enumerate(self._buffer[: self.max_chars]):
            length = index + 1
            if char in STRONG_BOUNDARIES:
                return index
            if char in SOFT_BOUNDARIES and length >= self.min_soft_chars:
                return index
        return None

    def _hard_split(self) -> int:
        prefix = self._buffer[: self.max_chars + 1]
        split_at = max(prefix.rfind(" "), prefix.rfind("\t"))
        return split_at if split_at > 0 else self.max_chars

    def _emit(self, chunks: list[str], end: int) -> None:
        chunk = self._buffer[:end].strip()
        self._buffer = self._buffer[end:].lstrip()
        if chunk:
            chunks.append(chunk)


def chunk_text(text: str, max_chars: int = 160) -> list[str]:
    chunker = IncrementalTextChunker(max_chars=max_chars)
    return chunker.feed(text) + chunker.flush()

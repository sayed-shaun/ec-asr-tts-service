"""In-memory cache of synthesized speech, keyed by what determines the audio.

Parler is autoregressive, so a reply costs seconds of GPU time and a worker
slot for as long as it decodes. Real TTS traffic is repetitive -- menu
prompts, confirmations, error messages, the same sentence read to every
caller -- and re-synthesizing that text is paying a GPU to reproduce bytes
the service already had.

Lives in the gateway rather than in the model worker because the gateway is
the single process every request passes through. LitServe runs a worker pool
and routes a request to whichever worker is free, so a cache down there would
be split across workers and miss most of the repeats it should catch.

Entries hold the per-clause PCM exactly as the worker streamed it, rather
than a finished WAV. That lets one entry serve every caller: `stream=true`
replays the clauses in order, `stream=false` joins them, and
`response_format="wav"` wraps the join in a header. Storing a WAV instead
would force the streaming path to re-split it, and storing per-format would
hold the same audio two or three times over.

One consequence worth stating: generation samples, so the same text
synthesized twice is already two slightly different recordings today. A hit
replays the one that was cached, which makes repeat requests consistent
rather than merely fast. That is usually what a caller wants; a caller who
wants a fresh take wants a cache miss, and nothing here offers one.

Per-process and unreplicated, so it is a latency and capacity optimization
and never a correctness guarantee -- a restart or a second gateway replica
just means more misses. Redis behind the same get/put pair is the change to
make if hit rate across replicas starts to matter.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import NamedTuple

_MIB = 1024 * 1024


class Speech(NamedTuple):
    """One cached reply: the clauses, in order, and the rate they play at."""

    sample_rate: int
    parts: tuple[bytes, ...]

    @property
    def nbytes(self) -> int:
        return sum(len(part) for part in self.parts)


class SpeechCache:
    """A byte-bounded LRU over synthesized replies.

    Bounded by total audio rather than by entry count because entry sizes
    vary by two orders of magnitude: at 44.1 kHz 16-bit mono a one-word reply
    is ~50 KB and a long paragraph is tens of MB, so any count that keeps
    short replies cheap lets long ones exhaust the gateway's memory.

    Locked rather than relying on the event loop being single-threaded:
    FastAPI runs sync dependencies and TestClient requests on a threadpool,
    and an OrderedDict reordered from two threads at once corrupts quietly.
    The critical sections are dict operations, so contention is not a concern.
    """

    def __init__(self, max_bytes: int, enabled: bool = True) -> None:
        self.max_bytes = max_bytes
        self.enabled = enabled
        self._entries: OrderedDict[str, Speech] = OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejections = 0

    @staticmethod
    def key(text: str, voice: str, description: str | None) -> str:
        """Hash the three inputs that decide what the audio sounds like.

        `response_format` and `stream` are deliberately absent: both are
        rendering decisions the router makes from the same stored PCM, so
        including them would store the same audio under several keys.

        Hashed rather than tupled so an entry costs a 64-character key
        instead of holding a caller's full 20,000-character input alive for
        as long as the audio it produced.

        Separated by a NUL, which the fields cannot contain, so that a text
        ending in a voice name cannot collide with the same text and that
        voice. Callers pass the voice already defaulted, so "" and the
        configured default share one entry instead of two.
        """
        parts = (text, voice, (description or "").strip())
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Speech | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: str, sample_rate: int, parts: list[bytes]) -> None:
        """Store a complete reply, evicting the coldest entries to fit.

        Callers must only reach here once the source stream is exhausted: a
        reply half-written because the client hung up would otherwise be
        served whole to everyone after it.
        """
        if not self.enabled:
            return
        entry = Speech(sample_rate=sample_rate, parts=tuple(parts))
        if entry.nbytes > self.max_bytes:
            self.rejections += 1
            return
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._nbytes -= existing.nbytes
            self._entries[key] = entry
            self._nbytes += entry.nbytes
            while self._nbytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._nbytes -= evicted.nbytes
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.rejections = 0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def stats(self) -> dict:
        """Counters for whoever is sizing the cap, reported in MiB.

        MiB rather than raw bytes because this is read by a human deciding
        whether TTS_CACHE_MAX_MB is too small, and that decision is never
        made to byte precision.
        """
        return {
            "enabled": self.enabled,
            "entries": len(self._entries),
            "used_mb": round(self._nbytes / _MIB, 1),
            "max_mb": round(self.max_bytes / _MIB, 1),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "rejections": self.rejections,
        }

    def sizing(self) -> str:
        """One line saying whether max_bytes is the wrong size, and which way.

        Evictions are the only honest signal for raising the cap: a cache
        that never evicts is already holding the entire working set, and
        giving it more memory cannot produce a hit it is not already getting.
        A low hit rate with no evictions means the text itself rarely repeats,
        which no cap fixes.
        """
        if not self.enabled:
            return "cache disabled"
        asked = self.hits + self.misses
        rate = f"{self.hits / asked:.1%}" if asked else "n/a"
        used = f"{self._nbytes / self.max_bytes:.0%}"
        verdict = (
            f"{self.evictions} evictions -- raising TTS_CACHE_MAX_MB may help"
            if self.evictions
            else "no evictions -- the working set fits, a larger cap buys nothing"
        )
        return f"hit rate {rate} over {asked} requests, {used} of cap used; {verdict}"

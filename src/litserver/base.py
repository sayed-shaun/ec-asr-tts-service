from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

NON_SPEECH_MAX_CHARS = 15
NON_SPEECH_MIN_CHAR_DENSITY = 3.0
NON_SPEECH_LITERALS = {"<>"}


@dataclass(frozen=True, slots=True)
class Audio:
    """One utterance: float32 mono samples plus their rate.

    Carries its own rate rather than assuming settings.SAMPLE_RATE, which is
    the input rate ASR expects; indic-parler-tts emits at 44.1kHz.
    """

    samples: np.ndarray
    sample_rate: int

    def pcm_s16le(self) -> bytes:
        """Return mono, signed 16-bit, little-endian PCM."""
        clipped = np.clip(self.samples, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2", copy=False).tobytes()


class BaseEngine(ABC):
    """What every model wrapper shares, whichever direction it runs in.

    Only load() is common to recognition and synthesis, so only it lives here;
    each task's own verb and helpers belong to the subclass. A LitAPI depends
    on one of those subclasses, never on this.

    Deliberately free of torch: an ASR engine can be pure ONNX, and importing
    torch here would put it on that path for nothing.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the checkpoint and run a warmup pass.

        Subclasses widen this with their own keyword arguments, all defaulted,
        so load() with no arguments stays valid for any engine.
        """


class BaseASREngine(BaseEngine):
    """Interface every ASR engine must implement.

    ASRLitAPI only calls load() and transcribe(); anything else is
    engine-specific and stays out of this contract. is_non_speech is shared
    rather than per-engine because the thresholds were measured once, on this
    corpus, and mean the same thing for any recognizer scored against it.
    """

    @abstractmethod
    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        """Return one transcript string per input audio array."""

    @staticmethod
    def is_non_speech(text: str, duration_seconds: float) -> bool:
        """True when `text` looks like a hallucination on non-speech audio.

        Known literals are flagged unconditionally; the density check cannot
        separate them from real speech on short segments. Anything else must be
        both tiny and sparse relative to its duration. Thresholds were measured
        against silence, noise, tone and music on this corpus.
        """
        stripped = "".join(text.split())
        if stripped in NON_SPEECH_LITERALS:
            return True
        chars = len(stripped)
        if chars == 0:
            return False
        if chars > NON_SPEECH_MAX_CHARS or duration_seconds <= 0:
            return False
        return chars / duration_seconds < NON_SPEECH_MIN_CHAR_DENSITY


class BaseTTSEngine(BaseEngine):
    """Interface every TTS engine must implement.

    TTS inverts the direction ASR runs in, so it shares load() and device
    resolution with BaseASREngine but none of the recognition-side helpers.
    """

    voices: dict[str, str] = {}
    """Named voices this engine accepts, mapped to their style description."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "",
        description: str | None = None,
    ) -> Audio:
        """Synthesize a single speakable chunk of text."""

    def speak_stream(
        self,
        text: str,
        voice: str = "",
        description: str | None = None,
    ) -> Iterator[Audio]:
        """Yield this request's audio piece by piece, in playback order.

        A caller that can play or send audio as it arrives should use this: on
        a live call it is the difference between first sound after the whole
        reply is synthesized and after the first clause. The default yields
        once; an engine that splits long input overrides it.
        """
        yield self.synthesize(text, voice, description)

    def speak(
        self,
        text: str,
        voice: str = "",
        description: str | None = None,
    ) -> Audio:
        """Turn one request's text into one Audio, however long it is.

        The batch counterpart of speak_stream, for callers that want a single
        buffer. Defined in terms of it so an engine only overrides one.
        """
        return self.join(list(self.speak_stream(text, voice, description)))

    @staticmethod
    def join(parts: list[Audio], gap_seconds: float = 0.08) -> Audio:
        """Concatenate per-chunk audio with a short silence between chunks.

        Chunks are cut on clause boundaries, so joining them with no gap
        sounds rushed at exactly the points a speaker would pause.
        """
        if not parts:
            raise ValueError("no audio was generated")
        sample_rate = parts[0].sample_rate
        if any(part.sample_rate != sample_rate for part in parts):
            raise ValueError("generated chunks have inconsistent sample rates")
        gap = np.zeros(round(gap_seconds * sample_rate), dtype=np.float32)
        arrays: list[np.ndarray] = []
        for index, part in enumerate(parts):
            if index:
                arrays.append(gap)
            arrays.append(part.samples)
        return Audio(np.concatenate(arrays), sample_rate)

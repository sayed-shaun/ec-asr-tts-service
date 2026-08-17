from abc import ABC, abstractmethod

import numpy as np

NON_SPEECH_MAX_CHARS = 15
NON_SPEECH_MIN_CHAR_DENSITY = 3.0
NON_SPEECH_LITERALS = {"<>"}


class BaseEngine(ABC):
    """Interface every ASR engine (Conformer, Wav2Vec2, ...) must implement.

    ASRLitAPI only calls load() and transcribe() — anything else is
    engine-specific and stays out of this contract.
    """

    @abstractmethod
    def load(self, warmup_seconds: float = 1.0, sample_rate: int = 16000) -> None:
        """Load the model/checkpoint and run a warmup inference."""

    @abstractmethod
    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        """Return one transcript string per input audio array."""

    @staticmethod
    def resolve_device(device: str) -> str:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def split(
        audio: np.ndarray,
        max_segment_samples: int,
        boundary_search_samples: int = 0,
    ) -> list[np.ndarray]:
        """Split audio longer than max_segment_samples into consecutive,
        non-overlapping chunks.

        Cuts are still non-overlapping (no overlap means no duplicated words in
        the joined transcript), but instead of slicing at exactly
        max_segment_samples, each boundary is pulled back to the quietest point
        within the preceding boundary_search_samples. A hard cut landing
        mid-word truncates it in both neighbouring segments, and CTC decoding
        of a clipped word tends to drop it entirely rather than emit half —
        losing real content silently. Cutting in a pause avoids that.

        boundary_search_samples=0 restores plain fixed-size cuts.
        """
        if len(audio) <= max_segment_samples:
            return [audio]

        # Must stay under a full segment so every cut still advances.
        search = min(boundary_search_samples, max_segment_samples // 2)

        segments = []
        start = 0
        while start < len(audio):
            end = start + max_segment_samples
            if end >= len(audio):
                segments.append(audio[start:])
                break
            cut = (
                BaseEngine.quietest_point(audio, end - search, end)
                if search > 0
                else end
            )
            segments.append(audio[start:cut])
            start = cut
        return segments

    @staticmethod
    def is_non_speech(text: str, duration_seconds: float) -> bool:
        """True when `text` looks like a hallucination on non-speech audio.

        Silence, noise and music all decode to a handful of stray characters
        instead of nothing (e.g. this repo's Whisper checkpoint decodes pure
        silence to the literal string "<>"). Known literals like that are
        flagged unconditionally -- on short segments the density check below
        can't tell "<>" apart from real speech (2 chars over ~0.5s is denser
        than NON_SPEECH_MIN_CHAR_DENSITY), so it must not gate them. Anything
        else still needs both thresholds to trip: the output has to be tiny
        *and* sparse relative to the audio it came from. See the module
        constants for the measured separation.
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

    @staticmethod
    def quietest_point(
        audio: np.ndarray, lo: int, hi: int, frame_samples: int = 160
    ) -> int:
        """Return the sample index of the lowest-energy frame in audio[lo:hi].

        Frames are 10ms at 16kHz — fine enough to land inside a short
        inter-word pause, coarse enough that a single quiet sample inside a
        voiced region can't win.

        Ties resolve to the *latest* quietest frame, which keeps segments as
        long as possible: on flat-energy audio (digital silence, a constant
        tone) every frame ties, so the cut stays at the far end of the search
        window instead of always jumping to its start.
        """
        lo = max(0, lo)
        window = audio[lo:hi]
        n_frames = window.size // frame_samples
        if n_frames < 2:
            return hi
        frames = window[: n_frames * frame_samples].reshape(n_frames, frame_samples)
        energy = np.abs(frames).mean(axis=1)
        latest_min = n_frames - 1 - int(np.argmin(energy[::-1]))
        return lo + latest_min * frame_samples + frame_samples // 2

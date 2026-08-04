import numpy as np
import torch
from loguru import logger

# Greedy CTC decoding on non-speech audio emits a few spurious characters
# rather than nothing ("তেন" for silence, "সগগগগগগ্গগ্গেন" for a music sting).
# Measured over the 1322-clip FLEURS benchmark, real speech never drops below
# 4.46 characters/second, while non-speech samples land between 0.07 and 2.80
# — and non-speech output is also tiny in absolute terms (1-14 characters).
# Requiring BOTH conditions is what makes this safe: a short but real phrase
# in a mostly-quiet segment exceeds the character cap and survives, so only
# "almost no output spread over a long stretch" is discarded.
NON_SPEECH_MAX_CHARS = 15
NON_SPEECH_MIN_CHAR_DENSITY = 3.0


class ASREngine:
    """Thin wrapper around the NeMo FastConformer CTC checkpoint
    (hishab/titu_stt_bn_fastconformer).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_segment_seconds: float = 18.0,
        boundary_search_seconds: float = 2.0,
        drop_non_speech: bool = True,
    ):
        self.model_name = model_name
        self.device = self.resolve_device(device)
        self.model = None
        self.max_segment_seconds = max_segment_seconds
        self.boundary_search_seconds = boundary_search_seconds
        self.drop_non_speech = drop_non_speech

    def load(self, warmup_seconds: float = 1.0, sample_rate: int = 16000) -> None:
        """Load the checkpoint and run a warmup inference.

        First inference pays a large one-off cost (cuDNN algo selection, CUDA
        kernel/JIT warmup, lazy module init) — measured ~740ms vs ~26ms steady
        state on an RTX 2050. Burn it here at startup so the first real
        request doesn't. Warmup failure is a warning, not fatal — a warmup
        that can't run must not stop the worker from coming up.
        """
        import nemo.collections.asr as nemo_asr

        logger.info(
            f"Loading NeMo ASR model '{self.model_name}' on device '{self.device}'"
        )
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )
        decoding_cfg = self.model.cfg.decoding
        decoding_cfg.strategy = "greedy_batch"
        self.model.change_decoding_strategy(decoding_cfg)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()

        if warmup_seconds > 0:
            try:
                silence = np.zeros(int(sample_rate * warmup_seconds), dtype=np.float32)
                self.transcribe([silence], batch_size=1, sample_rate=sample_rate)
                logger.info("ASR model warmed up")
            except Exception as exc:
                logger.warning(f"ASR model warmup failed (continuing anyway): {exc}")

        logger.info("ASR model loaded")

    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        if self.model is None:
            raise RuntimeError("ASREngine.load() must be called before transcribe()")

        max_segment_samples = int(sample_rate * self.max_segment_seconds)
        boundary_search_samples = int(sample_rate * self.boundary_search_seconds)
        segments_per_audio = [
            self.split(a, max_segment_samples, boundary_search_samples)
            for a in audios
        ]
        flat_segments = [seg for segments in segments_per_audio for seg in segments]

        with torch.inference_mode():
            hypotheses = self.model.transcribe(
                audio=flat_segments,
                batch_size=max(1, min(batch_size, len(flat_segments))),
                verbose=False,
            )
        texts = [self.as_text(h) for h in hypotheses]
        if self.drop_non_speech:
            texts = [
                ""
                if self.is_non_speech(text, len(segment) / sample_rate)
                else text
                for text, segment in zip(texts, flat_segments)
            ]

        results = []
        idx = 0
        for segments in segments_per_audio:
            n = len(segments)
            # Filter empties so a dropped segment doesn't leave double spaces.
            kept = [t for t in texts[idx : idx + n] if t]
            results.append(" ".join(kept).strip())
            idx += n
        return results

    @staticmethod
    def is_non_speech(text: str, duration_seconds: float) -> bool:
        """True when `text` looks like a hallucination on non-speech audio.

        Silence, noise and music all decode to a handful of stray characters
        instead of nothing. Both thresholds must trip: the output has to be
        tiny *and* sparse relative to the audio it came from. See the module
        constants for the measured separation.
        """
        chars = len("".join(text.split()))
        if chars == 0:
            return False
        if chars > NON_SPEECH_MAX_CHARS or duration_seconds <= 0:
            return False
        return chars / duration_seconds < NON_SPEECH_MIN_CHAR_DENSITY

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
                ASREngine.quietest_point(audio, end - search, end)
                if search > 0
                else end
            )
            segments.append(audio[start:cut])
            start = cut
        return segments

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

    @staticmethod
    def as_text(hypothesis) -> str:
        if isinstance(hypothesis, str):
            return hypothesis
        text = getattr(hypothesis, "text", None)
        return text if text is not None else str(hypothesis)

    @staticmethod
    def resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

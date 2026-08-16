import numpy as np
import torch
from loguru import logger

from src.litserver.engine.base import BaseEngine


class ConformerEngine(BaseEngine):
    """Thin wrapper around the NeMo FastConformer CTC checkpoint"""

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
            raise RuntimeError("ConformerEngine.load() must be called before transcribe()")

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
    def as_text(hypothesis) -> str:
        if isinstance(hypothesis, str):
            return hypothesis
        text = getattr(hypothesis, "text", None)
        return text if text is not None else str(hypothesis)

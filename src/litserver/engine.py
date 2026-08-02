import numpy as np
import torch
from loguru import logger


class ASREngine:
    """Thin wrapper around the NeMo FastConformer CTC checkpoint
    (hishab/titu_stt_bn_fastconformer).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_segment_seconds: float = 18.0,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.max_segment_seconds = max_segment_seconds

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
            f"Loading NeMo ASR model '{self.model_name}' on "
            f"device '{self.device}'"
        )
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            # Half precision roughly doubles throughput on GPU with
            # negligible accuracy impact for CTC models; not applied on CPU,
            # where most ops don't have an efficient fp16 kernel path.
            self.model = self.model.half()

        if warmup_seconds > 0:
            try:
                silence = np.zeros(
                    int(sample_rate * warmup_seconds), dtype=np.float32
                )
                self.transcribe([silence], batch_size=1, sample_rate=sample_rate)
                logger.info("ASR model warmed up")
            except Exception as exc:
                logger.warning(
                    f"ASR model warmup failed (continuing anyway): {exc}"
                )

        logger.info("ASR model loaded")

    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        if self.model is None:
            raise RuntimeError(
                "ASREngine.load() must be called before transcribe()"
            )

        max_segment_samples = int(sample_rate * self.max_segment_seconds)
        segments_per_audio = [self._split(a, max_segment_samples) for a in audios]
        flat_segments = [
            seg for segments in segments_per_audio for seg in segments
        ]

        with torch.inference_mode():
            hypotheses = self.model.transcribe(
                audio=flat_segments,
                batch_size=max(1, min(batch_size, len(flat_segments))),
                verbose=False,
            )
        texts = [self._as_text(h) for h in hypotheses]

        results = []
        idx = 0
        for segments in segments_per_audio:
            n = len(segments)
            results.append(" ".join(texts[idx : idx + n]).strip())
            idx += n
        return results

    @staticmethod
    def _split(audio: np.ndarray, max_segment_samples: int) -> list[np.ndarray]:
        """Split audio longer than max_segment_samples into consecutive,
        non-overlapping chunks (hard cuts, no overlap — simplest option that
        avoids duplicated words at chunk boundaries in the joined transcript).
        """
        if len(audio) <= max_segment_samples:
            return [audio]
        return [
            audio[start : start + max_segment_samples]
            for start in range(0, len(audio), max_segment_samples)
        ]

    @staticmethod
    def _as_text(hypothesis) -> str:
        if isinstance(hypothesis, str):
            return hypothesis
        text = getattr(hypothesis, "text", None)
        return text if text is not None else str(hypothesis)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

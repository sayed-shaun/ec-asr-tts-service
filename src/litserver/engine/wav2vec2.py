import numpy as np
import torch
from loguru import logger

from src.litserver.engine.base import BaseEngine


class Wav2Vec2Engine(BaseEngine):
    """Thin wrapper around a HuggingFace Wav2Vec2 CTC checkpoint."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_segment_seconds: float = 18.0,
        boundary_search_seconds: float = 2.0,
    ):
        self.model_name = model_name
        self.device = self.resolve_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.max_segment_seconds = max_segment_seconds
        self.boundary_search_seconds = boundary_search_seconds
        self.processor = None
        self.model = None

    def load(self, warmup_seconds: float = 1.0, sample_rate: int = 16000) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        logger.info(
            f"Loading Wav2Vec2 model '{self.model_name}' on device '{self.device}'"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCTC.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

        if warmup_seconds > 0:
            try:
                silence = np.zeros(int(sample_rate * warmup_seconds), dtype=np.float32)
                self.transcribe([silence], batch_size=1, sample_rate=sample_rate)
                logger.info("Wav2Vec2 model warmed up")
            except Exception as exc:
                logger.warning(f"Wav2Vec2 model warmup failed (continuing anyway): {exc}")

        logger.info("Wav2Vec2 model loaded")

    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        """Split long audio into segments before running inference.

        Self-attention memory scales quadratically with sequence length, so
        an unsegmented long clip can exhaust GPU memory well before any
        batching/concurrency limit kicks in — segmenting bounds the sequence
        length any single forward pass has to handle, same as ConformerEngine.
        """
        if self.model is None:
            raise RuntimeError("Wav2Vec2Engine.load() must be called before transcribe()")

        max_segment_samples = int(sample_rate * self.max_segment_seconds)
        boundary_search_samples = int(sample_rate * self.boundary_search_seconds)
        segments_per_audio = [
            self.split(a, max_segment_samples, boundary_search_samples)
            for a in audios
        ]
        flat_segments = [seg for segments in segments_per_audio for seg in segments]

        texts = []
        for start in range(0, len(flat_segments), max(1, batch_size)):
            batch = flat_segments[start : start + batch_size]
            inputs = self.processor(
                batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device, dtype=self.dtype)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.inference_mode():
                logits = self.model(
                    input_values, attention_mask=attention_mask
                ).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            texts.extend(self.processor.batch_decode(predicted_ids))

        results = []
        idx = 0
        for segments in segments_per_audio:
            n = len(segments)
            results.append(" ".join(t for t in texts[idx : idx + n] if t).strip())
            idx += n
        return results

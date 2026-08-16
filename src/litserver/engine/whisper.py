import numpy as np
import torch
from loguru import logger

from src.litserver.engine.base import BaseEngine


class WhisperEngine(BaseEngine):
    """Thin wrapper around a HuggingFace Whisper seq2seq checkpoint."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_segment_seconds: float = 28.0,
        boundary_search_seconds: float = 2.0,
        language: str = "bn",
        drop_non_speech: bool = True,
    ):
        self.model_name = model_name
        self.device = self.resolve_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.max_segment_seconds = max_segment_seconds
        self.boundary_search_seconds = boundary_search_seconds
        self.language = language
        self.drop_non_speech = drop_non_speech

        self.max_new_tokens = max(64, min(int(max_segment_seconds * 20), 448))
        self.processor = None
        self.model = None
        self.decoder_input_ids = None

    def load(self, warmup_seconds: float = 1.0, sample_rate: int = 16000) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        logger.info(
            f"Loading Whisper model '{self.model_name}' on device '{self.device}'"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)

        prompt_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )
        bos = self.model.config.decoder_start_token_id
        self.decoder_input_ids = torch.tensor(
            [[bos] + [token_id for _, token_id in prompt_ids]]
        )
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

        if warmup_seconds > 0:
            try:
                silence = np.zeros(int(sample_rate * warmup_seconds), dtype=np.float32)
                self.transcribe([silence], batch_size=1, sample_rate=sample_rate)
                logger.info("Whisper model warmed up")
            except Exception as exc:
                logger.warning(
                    f"Whisper model warmup failed (continuing anyway): {exc}"
                )

        logger.info("Whisper model loaded")

    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        """Whisper's feature extractor pads every segment to a fixed 30s
        log-mel window internally, so per-segment cost is constant
        regardless of actual length -- segmenting here exists to stay under
        that 30s model limit, not to bound variable memory the way
        Wav2Vec2Engine's segmenting does.
        """
        if self.model is None:
            raise RuntimeError(
                "WhisperEngine.load() must be called before transcribe()"
            )

        max_segment_samples = int(sample_rate * self.max_segment_seconds)
        boundary_search_samples = int(sample_rate * self.boundary_search_seconds)
        segments_per_audio = [
            self.split(a, max_segment_samples, boundary_search_samples) for a in audios
        ]
        flat_segments = [seg for segments in segments_per_audio for seg in segments]

        texts = []
        for start in range(0, len(flat_segments), max(1, batch_size)):
            batch = flat_segments[start : start + batch_size]
            inputs = self.processor(
                batch, sampling_rate=sample_rate, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device, dtype=self.dtype)

            decoder_input_ids = self.decoder_input_ids.expand(
                input_features.shape[0], -1
            ).to(self.device)
            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=self.max_new_tokens,
                )
            texts.extend(
                self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            )

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
            kept = [t.strip() for t in texts[idx : idx + n] if t.strip()]
            results.append(" ".join(kept).strip())
            idx += n
        return results

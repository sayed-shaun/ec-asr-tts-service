import numpy as np
from loguru import logger

from src.litserver.engine.base import BaseEngine


class ZipformerEngine(BaseEngine):
    """Thin wrapper around a sherpa-onnx streaming Zipformer2 transducer
    checkpoint (e.g. alphacep/vosk-model-small-streaming-bn).

    Unlike ConformerEngine/Wav2Vec2Engine/WhisperEngine, this model is
    designed to consume audio as a single growing stream rather than a fixed
    window, so there's no attention-memory blowup on long clips and no
    need for BaseEngine.split() here.

    Runs on the "cpu" onnxruntime execution provider regardless of
    settings.ACCELERATOR: the "cuda" provider needs a sherpa-onnx wheel built
    against a specific CUDA/cuDNN pair (see
    https://k2-fsa.github.io/sherpa/onnx/python/install.html), which isn't
    wired into this project's pip/Docker setup -- the plain PyPI wheel used
    here is CPU-only. This model is small enough that CPU inference is fine.
    """

    def __init__(
        self,
        model_name: str,
        num_threads: int = 2,
        decoding_method: str = "modified_beam_search",
        max_active_paths: int = 10,
        tail_padding_seconds: float = 1.0,
        drop_non_speech: bool = True,
    ):
        self.model_name = model_name
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self.tail_padding_seconds = tail_padding_seconds
        self.drop_non_speech = drop_non_speech
        self.recognizer = None

    def load(self, warmup_seconds: float = 1.0, sample_rate: int = 16000) -> None:
        import sherpa_onnx
        from huggingface_hub import hf_hub_download

        logger.info(f"Loading Zipformer model '{self.model_name}'")
        encoder = hf_hub_download(self.model_name, "am-onnx/encoder.onnx")
        decoder = hf_hub_download(self.model_name, "am-onnx/decoder.onnx")
        joiner = hf_hub_download(self.model_name, "am-onnx/joiner.onnx")
        tokens = hf_hub_download(self.model_name, "lang/tokens.txt")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=self.num_threads,
            sample_rate=sample_rate,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
            provider="cpu",
            model_type="zipformer2",
        )

        if warmup_seconds > 0:
            try:
                silence = np.zeros(int(sample_rate * warmup_seconds), dtype=np.float32)
                self.transcribe([silence], batch_size=1, sample_rate=sample_rate)
                logger.info("Zipformer model warmed up")
            except Exception as exc:
                logger.warning(f"Zipformer model warmup failed (continuing anyway): {exc}")

        logger.info("Zipformer model loaded")

    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        """Feed each clip into its own stream and decode streams together in
        batches -- sherpa-onnx's decode_streams() runs one encoder forward
        pass across all of them, same intent as the torch engines' batching.
        """
        if self.recognizer is None:
            raise RuntimeError("ZipformerEngine.load() must be called before transcribe()")

        tail_padding = np.zeros(
            int(sample_rate * self.tail_padding_seconds), dtype=np.float32
        )

        results = []
        for start in range(0, len(audios), max(1, batch_size)):
            batch = audios[start : start + batch_size]
            streams = []
            for audio in batch:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(sample_rate, audio)
                stream.accept_waveform(sample_rate, tail_padding)
                stream.input_finished()
                streams.append(stream)

            while any(self.recognizer.is_ready(s) for s in streams):
                ready = [s for s in streams if self.recognizer.is_ready(s)]
                self.recognizer.decode_streams(ready)

            for stream, audio in zip(streams, batch):
                text = self.recognizer.get_result(stream)
                if self.drop_non_speech and self.is_non_speech(
                    text, len(audio) / sample_rate
                ):
                    text = ""
                results.append(text)

        return results

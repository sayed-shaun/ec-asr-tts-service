from importlib.metadata import PackageNotFoundError, version

import numpy as np
import sherpa_onnx
from huggingface_hub import hf_hub_download
from loguru import logger

from src.core.config import settings
from src.litserver.base import BaseASREngine
from src.litserver.zipformer.layouts import DEFAULT, ZipformerLayout


def build(device: str) -> "ZipformerEngine":
    """Construct the engine from settings. device is ignored: onnxruntime
    picks its own provider, set by ZIPFORMER_PROVIDER rather than by the
    accelerator LitServe hands the worker."""
    return ZipformerEngine(
        model_name=settings.ZIPFORMER_MODEL_NAME,
        provider=settings.ZIPFORMER_PROVIDER,
    )


class ZipformerEngine(BaseASREngine):
    """Thin wrapper around a sherpa-onnx streaming Zipformer2 transducer.

    Consumes audio as one growing stream rather than a fixed window, so there
    is no attention-memory blowup on long clips and no need for
    BaseASREngine.split().

    The onnxruntime execution provider is ZIPFORMER_PROVIDER, not
    settings.ACCELERATOR: it is a property of the sherpa-onnx wheel, not of
    what LitServe hands the worker. "cuda" needs a wheel built against a
    matching CUDA/cuDNN pair from k2-fsa's index; the PyPI wheel is CPU-only
    and silently falls back, so a wrong setting costs performance rather than
    failing loudly.

    "cpu" is the default deliberately. The checkpoint is small and decodes a
    few hundred ms per forward pass, where kernel-launch overhead can exceed
    the compute, and the GPU is wanted for the autoregressive TTS model whose
    latency a caller actually hears. CUDA is worth measuring on the batch
    path, where decode_streams() runs one encoder pass over many utterances.

    Which files to download is a property of the checkpoint, not of this
    engine, so it comes in as a ZipformerLayout from layouts.py alongside
    this module. Serving a different transducer means passing another layout
    from there; nothing in this class changes.
    """

    def __init__(
        self,
        model_name: str,
        num_threads: int = 2,
        decoding_method: str = "modified_beam_search",
        max_active_paths: int = 10,
        tail_padding_seconds: float = 1.0,
        drop_non_speech: bool = True,
        layout: ZipformerLayout = DEFAULT,
        provider: str = "cpu",
        enable_endpoint_detection: bool = True,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
    ):
        self.model_name = model_name
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self.tail_padding_seconds = tail_padding_seconds
        self.drop_non_speech = drop_non_speech
        self.layout = layout
        self.provider = provider
        self.enable_endpoint_detection = enable_endpoint_detection
        self.rule1_min_trailing_silence = rule1_min_trailing_silence
        self.rule2_min_trailing_silence = rule2_min_trailing_silence
        self.rule3_min_utterance_length = rule3_min_utterance_length
        self.recognizer = None

    @staticmethod
    def wheel_supports_cuda() -> bool:
        """True when the installed sherpa-onnx wheel was built with CUDA.

        The CUDA wheels carry it in their local version
        ("1.13.5+cuda12.cudnn9..."); the PyPI wheel is a plain version and
        bundles a CPU-only onnxruntime.
        """
        try:
            return "+cuda" in version("sherpa-onnx")
        except PackageNotFoundError:
            return False

    def load(self, warmup_seconds: float = 1.0, sample_rate: int = 16000) -> None:
        if self.provider == "cuda" and not self.wheel_supports_cuda():
            logger.warning(
                "ZIPFORMER_PROVIDER=cuda but the installed sherpa-onnx wheel is "
                "CPU-only, so decoding will silently run on CPU. Rebuild with "
                "SHERPA_ONNX_CUDA_VERSION set to a wheel matching this image's "
                "CUDA/cuDNN (see https://k2-fsa.github.io/sherpa/onnx/cuda.html)."
            )
        logger.info(
            f"Loading Zipformer model '{self.model_name}' "
            f"(provider={self.provider})"
        )
        encoder = hf_hub_download(self.model_name, self.layout.encoder)
        decoder = hf_hub_download(self.model_name, self.layout.decoder)
        joiner = hf_hub_download(self.model_name, self.layout.joiner)
        tokens = hf_hub_download(self.model_name, self.layout.tokens)

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=self.num_threads,
            sample_rate=sample_rate,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
            provider=self.provider,
            model_type="zipformer2",
            enable_endpoint_detection=self.enable_endpoint_detection,
            rule1_min_trailing_silence=self.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.rule2_min_trailing_silence,
            rule3_min_utterance_length=self.rule3_min_utterance_length,
        )

        if warmup_seconds > 0:
            try:
                silence = np.zeros(int(sample_rate * warmup_seconds), dtype=np.float32)
                self.transcribe([silence], batch_size=1, sample_rate=sample_rate)
                logger.info("Zipformer model warmed up")
            except Exception as exc:
                logger.warning(f"Zipformer model warmup failed (continuing anyway): {exc}")

        logger.info("Zipformer model loaded")

    def stream(self) -> "ZipformerSession":
        """Open a live session that survives across many audio frames.

        transcribe() below is the batch path: it opens a stream, feeds it one
        clip and closes it. A call has no such boundary, so a voicebot holds
        one session for the whole turn and feeds it frames as they arrive.
        """
        if self.recognizer is None:
            raise RuntimeError("ZipformerEngine.load() must be called before stream()")
        return ZipformerSession(self.recognizer)

    def transcribe(
        self,
        audios: list[np.ndarray],
        batch_size: int = 4,
        sample_rate: int = 16000,
    ) -> list[str]:
        """Feed each clip into its own stream, decoding streams in batches.

        sherpa-onnx's decode_streams() runs one encoder forward pass across
        all of them, same intent as the torch engines' batching.
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


class ZipformerSession:
    """One caller's live audio stream, decoded incrementally.

    Holds a single sherpa-onnx stream open across frames, so decoding is
    incremental rather than re-running the whole buffer each time. Frames are
    fed in as they arrive and the partial transcript grows; when
    is_endpoint() reports the caller has stopped, the text so far is the final
    turn and reset() starts the next one on the same stream.
    """

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.stream = recognizer.create_stream()

    def accept(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """Feed one frame, decode what is ready, return the partial transcript."""
        self.stream.accept_waveform(sample_rate, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return self.text

    @property
    def text(self) -> str:
        return self.recognizer.get_result(self.stream)

    def is_endpoint(self) -> bool:
        """True once trailing silence says the caller finished a turn."""
        return self.recognizer.is_endpoint(self.stream)

    def reset(self) -> None:
        """Clear the transcript and start the next turn on the same stream."""
        self.recognizer.reset(self.stream)

    def finish(
        self, tail_padding_seconds: float = 1.0, sample_rate: int = 16000
    ) -> str:
        """Flush the tail and return the last transcript.

        The decoder lags its input, so without the padding the final word of a
        turn can still be undelivered when the caller hangs up.
        """
        padding = np.zeros(int(sample_rate * tail_padding_seconds), dtype=np.float32)
        self.stream.accept_waveform(sample_rate, padding)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return self.text

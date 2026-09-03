from collections.abc import Iterator

import numpy as np
import torch
from loguru import logger
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

from src.core.config import settings
from src.litserver.base import Audio, BaseTTSEngine
from src.litserver.parler.chunking import chunk_text
from src.litserver.parler.voices import DEFAULT_VOICE, VOICES


def build(device: str) -> "ParlerTTSEngine":
    """Construct the engine from settings. Kept beside the class so litapi
    holds no per-model constructor knowledge."""
    return ParlerTTSEngine(
        model_name=settings.TTS_MODEL_NAME,
        device=device,
        default_voice=settings.TTS_VOICE,
        max_chars=settings.TTS_MAX_CHARS,
    )


class ParlerTTSEngine(BaseTTSEngine):
    """Wrapper around ai4bharat/indic-parler-tts.

    Prompted twice: a description (the voice's style, from voices.py) drives
    the text encoder, and the prompt is the text to speak. A voice is
    therefore a natural-language string, not a speaker embedding, so swapping
    voices needs no extra checkpoint.

    Holds no lock around generate(): each LitServe worker is its own process
    calling predict() serially, so the model is never entered concurrently.
    """

    voices = VOICES

    @staticmethod
    def resolve_device(device: str) -> str:
        """Turn "auto" into a concrete device.

        Lives here rather than on BaseEngine because it needs torch, and this
        is the only engine that does; the ASR side is pure ONNX.
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        default_voice: str = DEFAULT_VOICE,
        max_chars: int = 160,
    ):
        self.model_name = model_name
        self.device = self.resolve_device(device)
        self.default_voice = default_voice
        self.max_chars = max_chars
        self.sample_rate = 0
        self.model = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None

    def load(
        self,
        warmup_text: str = "সংক্ষিপ্ত পরীক্ষা।",
        sample_rate: int | None = None,
    ) -> None:
        """Load the checkpoint and warm it up.

        bf16 on GPU, since this is a ~2.6GB checkpoint sharing the device with
        the ASR workers and bf16 keeps fp32's exponent range. CPU stays fp32,
        where bf16 matmuls are unaccelerated and end up slower.

        The description is tokenized by the text encoder's own tokenizer, a
        different checkpoint with a different vocabulary; reusing the prompt
        tokenizer silently mistokenizes the style text.
        """
        logger.info(f"Loading Parler TTS model '{self.model_name}' on {self.device}")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype
        )
        self.model = model.to(self.device).eval()
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path
        )
        self.sample_rate = int(model.config.sampling_rate)

        if warmup_text:
            try:
                self.synthesize(warmup_text)
                logger.info("Parler TTS model warmed up")
            except Exception as exc:
                logger.warning(f"Parler TTS warmup failed (continuing anyway): {exc}")

        logger.info(f"Parler TTS model loaded (sample_rate={self.sample_rate})")

    def synthesize(
        self,
        text: str,
        voice: str = "",
        description: str | None = None,
    ) -> Audio:
        """Synthesize one chunk, peak-normalized to -0.45dBFS.

        Parler's output level swings widely with the style prompt, so without
        normalizing, consecutive chunks of one response differ audibly in
        loudness.
        """
        if self.model is None:
            raise RuntimeError(
                "ParlerTTSEngine.load() must be called before synthesize()"
            )

        voice = voice or self.default_voice
        if voice not in self.voices:
            raise ValueError(f"unknown voice: {voice}")
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        style = (
            description.strip()
            if description and description.strip()
            else self.voices[voice]
        )
        with torch.inference_mode():
            desc = self.description_tokenizer(style, return_tensors="pt").to(
                self.device
            )
            prompt = self.prompt_tokenizer(text, return_tensors="pt").to(self.device)
            output = self.model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
            )
            samples = output[0].to(torch.float32).cpu().numpy().squeeze()

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(samples), initial=0.0))
        if peak > 1e-6:
            samples = samples * (0.95 / peak)
        return Audio(samples=samples, sample_rate=self.sample_rate)

    def speak_stream(
        self,
        text: str,
        voice: str = "",
        description: str | None = None,
    ) -> Iterator[Audio]:
        """Split into clauses and yield each as it finishes.

        Overridden because this model is autoregressive over a single prompt:
        long input degrades prosody and truncates past its token budget, so a
        request has to be cut up before it reaches synthesize(). Yielding as
        each clause completes is what lets a caller start playback during
        generation instead of after it.
        """
        for chunk in chunk_text(text, max_chars=self.max_chars):
            yield self.synthesize(chunk, voice, description)

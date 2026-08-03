import numpy as np


class SpeakerEncoder:
    """Wraps Resemblyzer's VoiceEncoder to produce L2-normalized speaker
    embeddings.

    Lazy-loaded on first embed() call rather than an explicit setup() step:
    this runs in LitServe's main process (see src/api/speaker/router.py), not a GPU
    worker, and the model is small enough (~17MB LSTM) that loading it
    on-demand is simpler than wiring a separate startup hook for one route.
    """

    def __init__(self):
        self.voice_encoder = None

    def embed(self, waveform: np.ndarray, sample_rate: int) -> list[float]:
        from resemblyzer import VoiceEncoder, preprocess_wav

        if self.voice_encoder is None:
            self.voice_encoder = VoiceEncoder()

        wav = preprocess_wav(waveform, source_sr=sample_rate)
        embedding = self.voice_encoder.embed_utterance(wav)
        return embedding.tolist()

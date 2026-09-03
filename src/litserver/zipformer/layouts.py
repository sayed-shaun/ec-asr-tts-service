"""Repo layouts for sherpa-onnx streaming Zipformer2 checkpoints.

Every published transducer names its four ONNX artifacts differently, and the
names are a property of the checkpoint, not of ZipformerEngine. Add a layout
here to serve a new one.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ZipformerLayout:
    """Where a checkpoint's four artifacts live inside its Hub repo."""

    encoder: str
    decoder: str
    joiner: str
    tokens: str


VOSK_BN = ZipformerLayout(
    encoder="am-onnx/encoder.onnx",
    decoder="am-onnx/decoder.onnx",
    joiner="am-onnx/joiner.onnx",
    tokens="lang/tokens.txt",
)
"""alphacep/vosk-model-small-streaming-bn, this project's default."""

K2_FSA = ZipformerLayout(
    encoder="encoder-epoch-99-avg-1.onnx",
    decoder="decoder-epoch-99-avg-1.onnx",
    joiner="joiner-epoch-99-avg-1.onnx",
    tokens="tokens.txt",
)
"""The k2-fsa/sherpa-onnx-streaming-zipformer-* family: flat repo, epoch-tagged
filenames. Untested here, but the layout the upstream docs publish."""

DEFAULT = VOSK_BN

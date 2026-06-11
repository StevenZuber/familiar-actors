"""Embed user-uploaded photos at request time via an ONNX export of CLIP.

The serving environment intentionally does not ship torch/open-clip (those
live in the `pipeline` dependency group). Instead, scripts/export_onnx.py
exports the ViT-B-32 image encoder to ONNX once, and this module replicates
open_clip's image preprocessing in PIL/numpy and runs the encoder with
onnxruntime. The resulting vectors live in the same embedding space as the
precomputed actor index, so cosine similarity against it is valid.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from familiar_actors.config import settings

logger = logging.getLogger(__name__)

# Preprocessing constants from open_clip's ViT-B-32 validation transform.
# These must match the transforms used to build the actor index, or
# similarity scores against it are meaningless.
IMAGE_SIZE = 224
_OPENAI_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_OPENAI_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Lazy-loaded onnxruntime session singleton. Typed Any because onnxruntime
# is an optional import here (the error surfaces as QueryEmbeddingUnavailable).
_session: Any = None


class QueryEmbeddingUnavailable(RuntimeError):
    """The ONNX image encoder (or onnxruntime) is not available."""


def onnx_model_path() -> Path:
    return settings.data_dir / "clip_image_encoder.onnx"


def _get_session() -> Any:
    global _session
    if _session is None:
        try:
            import onnxruntime  # type: ignore[import-untyped]
        except ImportError as e:
            raise QueryEmbeddingUnavailable(
                "onnxruntime is not installed; photo search is disabled."
            ) from e

        model_path = onnx_model_path()
        if not model_path.exists():
            raise QueryEmbeddingUnavailable(
                f"ONNX image encoder not found at {model_path}. "
                "Run scripts/export_onnx.py (requires the pipeline group) "
                "or include it in the deployed data release."
            )

        logger.info(f"Loading ONNX image encoder from {model_path}...")
        _session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        logger.info("ONNX image encoder loaded")
    return _session


def preprocess(image: Image.Image) -> np.ndarray:
    """Replicate open_clip's ViT-B-32 validation transform in PIL/numpy.

    Bicubic-resize so the short side is 224 (long side truncated like
    torchvision's Resize), center-crop to 224x224, scale to [0, 1], and
    normalize with the OpenAI CLIP mean/std. Returns a (1, 3, 224, 224)
    float32 array ready for the ONNX encoder.
    """
    image = image.convert("RGB")

    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = IMAGE_SIZE, int(IMAGE_SIZE * long / short)
    new_width, new_height = (
        (new_short, new_long) if width <= height else (new_long, new_short)
    )
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    left = int(round((new_width - IMAGE_SIZE) / 2.0))
    top = int(round((new_height - IMAGE_SIZE) / 2.0))
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - _OPENAI_MEAN) / _OPENAI_STD
    # HWC -> NCHW
    return pixels.transpose(2, 0, 1)[np.newaxis, :]


def embed_image(image: Image.Image) -> np.ndarray:
    """Generate a 512-d CLIP embedding for a PIL image.

    Raises QueryEmbeddingUnavailable if the ONNX encoder can't be loaded.
    The returned vector is unnormalized, matching what the pipeline stores;
    SimilarityIndex.search_by_vector normalizes queries itself.
    """
    session = _get_session()
    tensor = preprocess(image)
    (embedding,) = session.run(None, {"image": tensor})
    return embedding[0]

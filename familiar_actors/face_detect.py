"""Request-time face detection for the upload path (serving side).

Uses OpenCV's YuNet detector (a tiny ~230KB ONNX model shipped in the repo) so
production stays lean — no torch, no insightface. The pipeline detects with
buffalo_l/SCRFD instead, but that's fine: both feed the framing-robust CLIP
encoder and produce near-identical embeddings (verified crop-parity cosine
~0.99), and they share crop_to_face so the geometry is identical.

YuNet returns boxes as (x, y, w, h); we convert to (x1, y1, x2, y2) for
crop_to_face. Lazy-loaded singleton.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from familiar_actors.face_crop import crop_to_face

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet.onnx"
_detector: Any = None


def _get_detector() -> Any:
    global _detector
    if _detector is None:
        import cv2  # type: ignore[import-untyped]

        if not _MODEL_PATH.exists():
            raise RuntimeError(f"YuNet model not found at {_MODEL_PATH}")
        logger.info("Loading YuNet face detector...")
        # Input size is set per-image in detect_and_crop; (320, 320) is a placeholder.
        _detector = cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (320, 320))
        logger.info("YuNet face detector loaded")
    return _detector


def detect_and_crop(image: Image.Image) -> Image.Image | None:
    """Detect the largest face in a PIL image and return a padded face crop.

    Returns None if no face is detected (the caller surfaces a friendly
    message). The crop is padded via crop_to_face so hair/jaw/head-shape stay
    in frame, matching how the index embeddings were built.
    """
    import cv2  # type: ignore[import-untyped]

    detector = _get_detector()
    # PIL (RGB) -> OpenCV (BGR ndarray)
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None

    # YuNet rows are [x, y, w, h, <5 landmarks>, score]; pick the largest box.
    box = max(faces, key=lambda r: r[2] * r[3])
    x, y, bw, bh = box[:4]
    crop = crop_to_face(bgr, (x, y, x + bw, y + bh))
    if crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

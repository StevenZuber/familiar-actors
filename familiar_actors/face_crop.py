"""Shared face-crop geometry.

The pipeline (insightface detector) and the serving/upload path (a leaner
detector) use different detectors, but they MUST crop identically so the
stored face-crop embeddings and the query embeddings land in the same CLIP
space. That shared step is this module: given an image and a detected face
box, expand the box by a padding fraction and return the crop.

Pure numpy — no torch, onnx, or opencv — so it's safe to import anywhere.
"""

import numpy as np

from familiar_actors.config import settings


def crop_to_face(
    image: np.ndarray, bbox: tuple[float, float, float, float], pad: float | None = None
) -> np.ndarray:
    """Crop `image` (H, W, C) to the face `bbox` (x1, y1, x2, y2), padded.

    The box is expanded by `pad` (default settings.face_crop_pad) of its own
    width/height on each side, then clamped to the image bounds — so hair,
    jawline and head shape stay in frame, which matter for perceived likeness.
    Returns the cropped sub-image (a view); may be empty if the box is degenerate.
    """
    if pad is None:
        pad = settings.face_crop_pad

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    cx1 = max(0, int(round(x1 - pad * bw)))
    cy1 = max(0, int(round(y1 - pad * bh)))
    cx2 = min(w, int(round(x2 + pad * bw)))
    cy2 = min(h, int(round(y2 + pad * bh)))

    return image[cy1:cy2, cx1:cx2]


def largest_face(faces: list) -> object | None:
    """Pick the largest-area face from a list of detections with a `.bbox`.

    Uploads and headshots can contain several faces (group shots, posters);
    the subject is almost always the biggest one.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

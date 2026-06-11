"""Export the CLIP image encoder to ONNX for request-time photo search.

The web app embeds user-uploaded photos with onnxruntime instead of
shipping torch/open-clip. This script exports the image encoder from the
exact checkpoint used to build the actor index (settings.embedding_model /
settings.clip_pretrained), then verifies that ONNX inference with the
app's PIL/numpy preprocessing matches torch within a tight tolerance.

The output (data/clip_image_encoder.onnx) must be included in the
deployment data release tarball alongside the consolidated index.

Usage:
    uv run python scripts/export_onnx.py  (requires: uv sync --group pipeline)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from familiar_actors.config import settings
from familiar_actors.query_embedding import onnx_model_path, preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _verification_image():
    """Use a real headshot if one exists locally, else synthetic noise."""
    from PIL import Image

    headshots = sorted(settings.headshots_dir.glob("*.jpg"))
    if headshots:
        logger.info(f"Verifying against {headshots[0]}")
        return Image.open(headshots[0])

    logger.info("No local headshots; verifying against a synthetic image")
    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 255, (320, 280, 3), dtype=np.uint8))


def _verify(model, clip_preprocess, out_path: Path) -> None:
    import onnxruntime  # type: ignore[import-untyped]

    image = _verification_image()

    with torch.no_grad():  # type: ignore[no-untyped-call]
        torch_vec = (
            model.encode_image(clip_preprocess(image.convert("RGB")).unsqueeze(0))
            .squeeze()
            .numpy()
        )

    session = onnxruntime.InferenceSession(
        str(out_path), providers=["CPUExecutionProvider"]
    )
    (onnx_out,) = session.run(None, {"image": preprocess(image)})
    onnx_vec = onnx_out[0]

    cosine = float(
        np.dot(torch_vec, onnx_vec)
        / (np.linalg.norm(torch_vec) * np.linalg.norm(onnx_vec))
    )
    logger.info(f"torch vs ONNX cosine similarity: {cosine:.6f}")
    if cosine < 0.999:
        raise SystemExit(
            f"Parity check failed (cosine {cosine:.6f} < 0.999). "
            "Preprocessing or export does not match the torch pipeline."
        )
    logger.info("Parity check passed")


def main():
    import open_clip  # type: ignore[import-untyped]

    logger.info(f"Loading {settings.embedding_model} ({settings.clip_pretrained})...")
    model, _, clip_preprocess = open_clip.create_model_and_transforms(
        settings.embedding_model,
        pretrained=settings.clip_pretrained,
    )
    model.eval()

    out_path = onnx_model_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting image encoder to {out_path}...")
    torch.onnx.export(
        model.visual,
        (torch.randn(1, 3, 224, 224),),
        str(out_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    logger.info(f"Exported: {out_path.stat().st_size / 1024 / 1024:.1f}MB")

    _verify(model, clip_preprocess, out_path)


if __name__ == "__main__":
    main()

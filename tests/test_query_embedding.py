import numpy as np
import pytest
from PIL import Image

from familiar_actors import query_embedding
from familiar_actors.query_embedding import (
    IMAGE_SIZE,
    QueryEmbeddingUnavailable,
    preprocess,
)


@pytest.mark.unit
class TestPreprocess:
    def test_output_shape_and_dtype(self):
        image = Image.new("RGB", (300, 400), color=(120, 80, 200))
        tensor = preprocess(image)
        assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert tensor.dtype == np.float32

    def test_handles_wide_tall_and_small_images(self):
        for size in [(640, 480), (480, 640), (100, 100), (224, 224)]:
            tensor = preprocess(Image.new("RGB", size))
            assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_converts_non_rgb_modes(self):
        tensor = preprocess(Image.new("L", (300, 300)))
        assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_normalization_applied(self):
        # A pure white image normalizes to (1 - mean) / std, all positive
        # and well above 1 — confirms scaling and mean/std are applied.
        tensor = preprocess(Image.new("RGB", (300, 300), color=(255, 255, 255)))
        assert tensor.min() > 1.0


@pytest.mark.unit
class TestGetSession:
    def test_missing_model_raises_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(query_embedding, "_session", None)
        monkeypatch.setattr(
            query_embedding.settings, "data_dir", tmp_path, raising=True
        )
        with pytest.raises(QueryEmbeddingUnavailable, match="not found"):
            query_embedding.embed_image(Image.new("RGB", (224, 224)))

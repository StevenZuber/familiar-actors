import numpy as np
import pytest
from PIL import Image

from familiar_actors import face_detect


@pytest.mark.unit
class TestDetectAndCrop:
    def test_no_face_returns_none(self):
        # A flat-color image has no detectable face.
        blank = Image.new("RGB", (320, 320), color=(120, 120, 120))
        assert face_detect.detect_and_crop(blank) is None

    def test_noise_image_returns_none(self):
        rng = np.random.default_rng(0)
        noise = Image.fromarray(rng.integers(0, 255, (240, 240, 3), dtype=np.uint8))
        # Random noise should not produce a confident face detection.
        assert face_detect.detect_and_crop(noise) is None

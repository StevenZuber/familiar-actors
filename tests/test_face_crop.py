import numpy as np
import pytest

from familiar_actors.face_crop import crop_to_face, largest_face


@pytest.mark.unit
class TestCropToFace:
    def test_crop_expands_box_by_padding(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # 20x20 box at (40,40)-(60,60); pad=0.5 -> expand 10px each side
        crop = crop_to_face(img, (40, 40, 60, 60), pad=0.5)
        assert crop.shape[:2] == (40, 40)

    def test_crop_clamps_to_image_bounds(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box near the edge; padding would go negative / past the edge
        crop = crop_to_face(img, (0, 0, 20, 20), pad=1.0)
        assert crop.shape[0] <= 100 and crop.shape[1] <= 100
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_zero_padding_returns_exact_box(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_to_face(img, (30, 25, 70, 85), pad=0.0)
        assert crop.shape[:2] == (60, 40)  # (y2-y1, x2-x1)

    def test_default_pad_comes_from_settings(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = crop_to_face(img, (80, 80, 120, 120))  # 40px box, default pad 0.35
        # 0.35 * 40 = 14px each side -> 40 + 28 = 68
        assert crop.shape[:2] == (68, 68)


@pytest.mark.unit
class TestLargestFace:
    def test_returns_none_for_empty(self):
        assert largest_face([]) is None

    def test_picks_biggest_box(self):
        class F:
            def __init__(self, bbox):
                self.bbox = bbox

        small = F((0, 0, 10, 10))
        big = F((0, 0, 50, 50))
        assert largest_face([small, big, small]) is big

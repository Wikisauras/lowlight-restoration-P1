"""Edge-case tests: run with  python -m pytest tests -q   (or python tests/test_edge_cases.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hybrid import enhance  # noqa: E402

MODES = ("classical", "dl", "hybrid", "auto")
rng = np.random.default_rng(0)


def _check(img, expect_shape=None, gray=False):
    for m in MODES:
        y, info = enhance(img, m)
        assert np.isfinite(y).all(), m
        assert y.min() >= 0 and y.max() <= 1, m
        h, w = img.shape[:2]
        assert y.shape == (h, w, 3), (m, y.shape)
        if gray:
            assert np.allclose(y[..., 0], y[..., 1]) and np.allclose(y[..., 1], y[..., 2]), m
    return y, info


def test_completely_black():
    img = np.zeros((120, 160, 3), np.uint8)
    y, info = _check(img)
    assert y.max() < 0.02 and info["chosen"] == "passthrough"


def test_near_black_noise():
    img = rng.integers(0, 3, (120, 160, 3)).astype(np.uint8)
    _check(img)


def test_saturated_bright_spots():
    img = (rng.random((200, 300, 3)) * 30).astype(np.uint8)
    img[50:80, 100:140] = 255
    y, _ = _check(img)
    assert y[55:75, 105:135].mean() > 0.95           # light source kept bright
    assert (y.max(axis=2) >= 0.99).mean() < 0.05      # no global blow-out


def test_tiny_image():
    _check((rng.random((8, 8, 3)) * 40).astype(np.uint8))
    _check((rng.random((1, 1, 3)) * 40).astype(np.uint8))
    _check((rng.random((5, 300, 3)) * 40).astype(np.uint8))


def test_huge_image():
    img = (rng.random((3000, 4000, 3)) * 40).astype(np.uint8)
    y, info = enhance(img, "hybrid")
    assert y.shape == (3000, 4000, 3)


def test_grayscale_inputs():
    g = (rng.random((100, 140)) * 40).astype(np.uint8)
    _check(g, gray=True)
    _check(g[..., None], gray=True)
    _check(np.repeat(g[..., None], 3, 2), gray=True)


def test_16bit_and_float_and_rgba():
    _check((rng.random((64, 64, 3)) * 4000).astype(np.uint16))
    _check((rng.random((64, 64, 3)) * 0.1).astype(np.float32))
    _check(np.concatenate([(rng.random((64, 64, 3)) * 40), np.full((64, 64, 1), 255)], 2).astype(np.uint8))


def test_already_bright_not_destroyed():
    img = (np.clip(rng.normal(150, 40, (128, 128, 3)), 0, 255)).astype(np.uint8)
    y, _ = enhance(img, "hybrid")
    assert abs(y.mean() - img.mean() / 255) < 0.12


if __name__ == "__main__":
    import time
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            t = time.time()
            f()
            print(f"PASS {k} ({time.time() - t:.1f}s)")

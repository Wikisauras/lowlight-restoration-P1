"""
make_figures.py - Report figures: intermediate representations and
non-uniform / backlit examples.

  python scripts/make_figures.py --out report/figures --images data/real_unseen/1021_0_.png data/synthetic/low/non_uniform__020.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import imread, imwrite, label_tile  # noqa: E402
from src.hybrid import enhance  # noqa: E402
from src.analysis import to_float_rgb  # noqa: E402


def cmap(m, c=cv2.COLORMAP_INFERNO):
    m = (m - m.min()) / max(m.max() - m.min(), 1e-6)
    return cv2.cvtColor(cv2.applyColorMap((m * 255).astype(np.uint8), c), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def intermediates_figure(path, out, h=230):
    img = imread(path)
    y, info = enhance(img, "hybrid")
    I = info["intermediates"]
    fw = I["fusion_weights"]
    wrgb = np.zeros(fw["classical"].shape + (3,), np.float32)
    wrgb[..., 2] = fw["classical"]
    wrgb[..., 0] = fw["dl_prior"]
    tiles = [
        ("1 input", to_float_rgb(img)[0]),
        ("2 illumination T (max-RGB+GF)", cmap(I["classical_illumination"])),
        ("3 reflectance R=I/T", I["classical_reflectance"]),
        ("4 Retinex relit + MSRCR", I["classical_retinex"]),
        ("5 guided-filter denoised", I["classical_denoised"]),
        ("6 adaptive CLAHE (L*)", I["classical_clahe"]),
        ("7 classical output", I["classical_final"]),
        ("8 Zero-DCE++ curve map |A|", cmap(np.abs(I["dl_curve_map"]).mean(2), cv2.COLORMAP_VIRIDIS)),
        ("9 illumination prior m", cmap(I["prior_modulation"], cv2.COLORMAP_VIRIDIS)),
        ("10 DL (prior) + denoise", I["dl_denoised"]),
        ("11 fusion weights (B=cls,R=DL)", wrgb),
        ("12 final hybrid", y),
    ]
    rows = []
    for i in range(0, 12, 4):
        r = []
        for lab, t in tiles[i:i + 4]:
            t = cv2.resize(t.astype(np.float32), (int(t.shape[1] * h / t.shape[0]), h))
            r.append(label_tile(t, lab, 0.45))
        rows.append(np.concatenate(r, 1))
    imwrite(out, np.concatenate(rows, 0))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--images", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in a.images:
        stem = os.path.splitext(os.path.basename(p))[0]
        info = intermediates_figure(p, os.path.join(a.out, f"intermediates_{stem}.png"))
        print(stem, info["stats"].condition, info["stats"].flags)


if __name__ == "__main__":
    main()

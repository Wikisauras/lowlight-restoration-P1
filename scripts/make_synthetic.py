"""
make_synthetic.py - Build a paired synthetic low-light test set with ground truth.

Degradation model (applied in linear light, then re-encoded to sRGB):
    I_low = Q( encode( P( e * L(x,y) * c * decode(I_gt) ) ) )
  decode/encode : sRGB <-> linear (gamma 2.2 approximation)
  e             : global exposure factor
  L(x,y)        : spatial illumination pattern (1 for uniform; spotlight for non-uniform)
  c             : per-channel colour-cast gains
  P             : Poisson-Gaussian sensor noise (shot noise ~ signal, plus read noise)
  Q             : 8-bit quantisation

Five illumination conditions (tests generalisation, requirement B3):
  dark, extreme_dark, non_uniform, colour_cast, bright_spots

Usage:
  python scripts/make_synthetic.py --src <folder of well-exposed images> --out data/synthetic --n 12
"""
import argparse
import glob
import os

import cv2
import numpy as np

CONDITIONS = ("dark", "extreme_dark", "non_uniform", "colour_cast", "bright_spots")


def srgb_to_lin(x):
    return np.power(np.clip(x, 0, 1), 2.2)


def lin_to_srgb(x):
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


def add_noise(lin, rng, shot, read):
    # Poisson-Gaussian: variance = shot*signal + read^2
    var = shot * lin + read ** 2
    return lin + rng.normal(0, 1, lin.shape).astype(np.float32) * np.sqrt(var)


def spotlight(h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    s = rng.uniform(0.18, 0.3) * max(h, w)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    return g


def add_emitters(img, rng, k=3):
    """Add k bright light-source discs (saturated) - identical in GT and low."""
    h, w = img.shape[:2]
    out = img.copy()
    masks = np.zeros((h, w), np.float32)
    for _ in range(k):
        cx, cy = int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h)
        r = int(rng.uniform(0.02, 0.05) * min(h, w)) + 2
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (cx, cy), r, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), r * 0.6)
        masks = np.maximum(masks, m / max(m.max(), 1e-6))
    col = np.array([1.0, 0.95, 0.85], np.float32)
    out = out * (1 - masks[..., None]) + masks[..., None] * col
    return np.clip(out, 0, 1), masks


def degrade(gt, cond, rng):
    h, w = gt.shape[:2]
    lin = srgb_to_lin(gt)
    if cond == "dark":
        e = rng.uniform(0.06, 0.12); L = 1.0; c = np.ones(3); shot, read = 0.002, 0.002
    elif cond == "extreme_dark":
        e = rng.uniform(0.003, 0.008); L = 1.0; c = np.ones(3); shot, read = 0.0012, 0.0015
    elif cond == "non_uniform":
        e = rng.uniform(0.02, 0.04); g = spotlight(h, w, rng)
        L = (g * (0.9 / e - 1) + 1)[..., None]   # peak region near full exposure
        c = np.ones(3); shot, read = 0.002, 0.002
    elif cond == "colour_cast":
        e = rng.uniform(0.06, 0.12); L = 1.0
        c = np.array([1.0, 0.62, 0.32]) if rng.uniform() < 0.5 else np.array([0.4, 0.7, 1.0])
        shot, read = 0.002, 0.002
    elif cond == "bright_spots":
        e = rng.uniform(0.04, 0.08); L = 1.0; c = np.ones(3); shot, read = 0.002, 0.002
    else:
        raise ValueError(cond)
    low = lin * e * L * c[None, None, :]
    low = add_noise(low.astype(np.float32), rng, shot * e, read)
    low = lin_to_srgb(low)
    if cond == "bright_spots":
        gt, m = add_emitters(gt, rng)
        low = np.clip(low * (1 - m[..., None]) + m[..., None] * gt, 0, 1)
    return gt, np.round(np.clip(low, 0, 1) * 255) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=12, help="images per condition")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.src, "*.jpg")) + glob.glob(os.path.join(a.src, "*.png")))
    # keep only well-exposed sources (mean 90..170 / 255) to serve as ground truth
    good = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        m = im.mean()
        if 90 < m < 170 and im.std() > 40:
            good.append(f)
        if len(good) >= a.n * len(CONDITIONS):
            break
    os.makedirs(os.path.join(a.out, "low"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "high"), exist_ok=True)
    k = 0
    for ci, cond in enumerate(CONDITIONS):
        for j in range(a.n):
            f = good[(ci * a.n + j) % len(good)]
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            im = cv2.resize(im, (a.size, a.size), interpolation=cv2.INTER_AREA)
            gt, low = degrade(im, cond, rng)
            name = f"{cond}__{k:03d}.png"
            cv2.imwrite(os.path.join(a.out, "high", name), cv2.cvtColor((gt * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(a.out, "low", name), cv2.cvtColor((low * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            k += 1
    print(f"wrote {k} pairs to {a.out}")


if __name__ == "__main__":
    main()

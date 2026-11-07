"""
classical.py - Classical, illumination-adaptive low-light enhancement.

Pipeline (each stage can be switched off for the ablation study):

  1. Gray-world colour correction         strength from analysis.colour_cast
  2. Illumination estimation              max-RGB initial map T0 (LIME, Guo et al. 2017)
  3. Illumination refinement              edge-aware guided filter (He et al. 2013)
  4. Retinex decomposition / relighting   R = I / T ;  relit = I / T^gamma  (gamma adaptive)
  5. MSRCR blend                          multi-scale Retinex with colour restoration
                                          (Jobson et al. 1997), weight adaptive
  6. Edge-preserving denoising            guided filter in YCrCb, eps tied to noise sigma,
                                          stronger in dark regions ((1-T)^p mask)
  7. Adaptive CLAHE                       on the CIELAB L channel, clip/tiles adaptive
  8. Highlight protection                 keep already well-lit pixels, soft-knee tone curve
"""
from __future__ import annotations

import cv2
import numpy as np

from .analysis import luminance, noise_sigma, EPS

DENOISE_AFTER_CLAHE = True

ALL_COMPONENTS = ("colour", "illumination", "msrcr", "denoise", "clahe", "highlight")


# ----------------------------------------------------------------------------
# Guided filter (own implementation; no opencv-contrib dependency)
# ----------------------------------------------------------------------------
def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Gray-guide guided filter. guide: HxW, src: HxW or HxWxC (filtered per channel)."""
    guide = guide.astype(np.float32)
    r = max(1, int(r))
    if src.ndim == 3:
        return np.stack([guided_filter(guide, src[..., c], r, eps) for c in range(src.shape[2])], axis=2)
    src = src.astype(np.float32)
    mI = _box(guide, r)
    mp = _box(src, r)
    cov = _box(guide * src, r) - mI * mp
    var = _box(guide * guide, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return _box(a, r) * guide + _box(b, r)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def gray_world(x: np.ndarray, strength: float) -> np.ndarray:
    """Gray-world white balance on reliable pixels, blended by `strength`."""
    if strength <= 0:
        return x
    y = luminance(x)
    valid = (y > 0.02) & (x.max(axis=2) < 0.98)
    if valid.sum() < 64:
        valid = np.ones_like(y, bool)
    means = np.array([x[..., c][valid].mean() for c in range(3)]) + EPS
    gains = np.clip(means.mean() / means, 0.6, 1.7)
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(x * gains[None, None, :].astype(np.float32), 0, 1)


def estimate_illumination(x: np.ndarray, radius: int, eps: float):
    """LIME-style illumination: max-RGB initial map refined by a guided filter.

    Returns (T0, T) both HxW in (0, 1].
    """
    t0 = x.max(axis=2).astype(np.float32)
    guide = luminance(x)
    t = guided_filter(guide, t0, radius, eps)
    # illumination must bound every channel from above: T >= max(R,G,B) (softly)
    t = np.maximum(t, 0.85 * t0)
    t = np.clip(t, 0.01, 1.0)
    return t0, t


def relight(x: np.ndarray, t: np.ndarray, gamma: float):
    """Retinex relighting. R = I/T (reflectance), output = I / T^gamma."""
    refl = np.clip(x / t[..., None], 0, 1)
    out = x / np.power(t, gamma)[..., None]
    return np.clip(out, 0, 1), refl


def msrcr(x: np.ndarray, sigmas, alpha=125.0, beta=46.0, G=192.0, b=-30.0):
    """Multi-Scale Retinex with Colour Restoration (Jobson, Rahman, Woodell 1997).

    Large-sigma Gaussians are computed on a downscaled copy for speed.
    Output is stretched with a 1%/99% simplest colour balance.
    """
    I = x.astype(np.float32) * 255.0 + 1.0
    logI = np.log(I)
    msr = np.zeros_like(I)
    h, w = x.shape[:2]
    for s in sigmas:
        f = 1.0
        if s > 30:
            f = max(1.0, s / 30.0)
        if f > 1.0 and min(h, w) / f >= 8:
            small = cv2.resize(I, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)
            blur = cv2.GaussianBlur(small, (0, 0), s / f)
            blur = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            blur = cv2.GaussianBlur(I, (0, 0), min(s, max(h, w)))
        msr += logI - np.log(np.maximum(blur, 1.0))
    msr /= len(sigmas)
    crf = beta * (np.log(alpha * I) - np.log(I.sum(axis=2, keepdims=True)))
    out = G * (msr * crf + b)
    # simplest colour balance per channel
    res = np.empty_like(out)
    for c in range(3):
        lo, hi = np.percentile(out[..., c], (1, 99))
        res[..., c] = (out[..., c] - lo) / max(hi - lo, EPS)
    return np.clip(res, 0, 1)


def match_brightness(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale src so its mean luminance equals ref's (used before blending)."""
    ys, yr = luminance(src).mean(), luminance(ref).mean()
    return np.clip(src * (yr / max(ys, EPS)), 0, 1)


def denoise(x: np.ndarray, t: np.ndarray, p: dict) -> np.ndarray:
    """Edge-preserving guided-filter denoising in YCrCb.

    eps is tied to the expected post-enhancement noise sigma (eps = (k*sigma)^2:
    structures with local variance well above the noise variance are preserved,
    flat noisy areas are averaged). Strong noise -> two guided-filter passes.
    Chroma is filtered with a larger window guided by the denoised luma.
    The result is blended back with weight derived from (1 - T)^(1/power) so dark
    regions (largest gain, largest noise amplification) are denoised the most.
    """
    if p["dn_strength"] <= 0.01:
        return x
    ycc = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    r = int(p["dn_radius"])
    y_d = guided_filter(y, y, r, p["dn_eps"])
    if p["dn_strength"] > 0.6:
        y_d = guided_filter(y_d, y_d, max(1, r // 2), p["dn_eps"] * 0.5)
    rc = int(max(2, 2 * r))
    cr_d = guided_filter(y_d, ycc[..., 1], rc, p["dn_chroma_eps"])
    cb_d = guided_filter(y_d, ycc[..., 2], rc, p["dn_chroma_eps"])
    den = cv2.cvtColor(np.stack([y_d, cr_d, cb_d], 2).astype(np.float32), cv2.COLOR_YCrCb2RGB)
    m = np.power(np.clip(1.0 - t, 0, 1), 1.0 / max(p["dn_dark_power"], 1e-3))
    m = np.clip(m * (0.5 + 0.5 * p["dn_strength"]) + 0.3 * p["dn_strength"], 0, 1)[..., None]
    return np.clip(m * den + (1 - m) * x, 0, 1)


def clahe_lab(x: np.ndarray, clip: float, tiles: int, blend: float = 1.0) -> np.ndarray:
    """CLAHE on the L channel of CIELAB (16-bit for precision)."""
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    L = lab[..., 0] / 100.0
    tiles = max(1, int(tiles))
    h, w = L.shape
    tiles_h = max(1, min(tiles, h // 8))
    tiles_w = max(1, min(tiles, w // 8))
    cl = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tiles_w, tiles_h))
    L16 = np.clip(L * 65535, 0, 65535).astype(np.uint16)
    Lc = cl.apply(L16).astype(np.float32) / 65535.0
    lab[..., 0] = 100.0 * (blend * Lc + (1 - blend) * L)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)


def soft_knee(y: np.ndarray, knee: float) -> np.ndarray:
    """Identity below `knee`, smooth tanh roll-off above it (never exceeds 1)."""
    k = float(knee)
    out = y.copy()
    hi = y > k
    out[hi] = k + (1 - k) * np.tanh((y[hi] - k) / max(1 - k, EPS))
    return out


def protect_highlights(out: np.ndarray, x_in: np.ndarray, p: dict) -> np.ndarray:
    """(a) a soft knee on the luminance of the ENHANCED content (RGB scaled to keep
    hue) rolls off values that the gain pushed towards clipping;
    (b) pixels that were already well lit in the input are kept close to their
    input value (never darker, at most slightly brighter) - this part is NOT
    knee-compressed, so genuine light sources stay at full brightness."""
    y_in = luminance(x_in)
    lo = p["protect_lo"]
    m = np.clip((y_in - lo) / max(1 - lo, EPS), 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)[..., None]
    y = luminance(out)
    y2 = soft_knee(y, p["knee"])
    kneed = np.clip(out * (y2 / np.maximum(y, EPS))[..., None], 0, 1)
    keep = np.clip(np.maximum(x_in, np.minimum(out, x_in * 1.1 + 0.02)), 0, 1)
    return np.clip(m * keep + (1 - m) * kneed, 0, 1)


# ----------------------------------------------------------------------------
# Full classical pipeline
# ----------------------------------------------------------------------------
def enhance_classical(x: np.ndarray, p: dict, components=ALL_COMPONENTS, return_intermediates=True):
    """Run the classical pipeline. x: float RGB [0,1]; p: dict from select_params."""
    comps = set(components)
    inter = {}
    x0 = x
    # 1. colour correction
    xc = gray_world(x, p["cc_strength"]) if "colour" in comps else x
    inter["colour_corrected"] = xc

    # 2-3. illumination
    t0, t = estimate_illumination(xc, p["gf_radius"], p["gf_eps"])
    inter["illumination_init"] = t0
    inter["illumination"] = t

    # 4. relighting
    if "illumination" in comps:
        relit, refl = relight(xc, t, p["gamma"])
    else:
        # without illumination decomposition: a global gamma reaching the same mean
        ym = max(luminance(xc).mean(), 1e-3)
        g = np.log(0.45) / np.log(ym) if ym < 0.45 else 1.0
        relit = np.power(np.clip(xc, 0, 1), np.clip(g, 0.2, 1.0))
        refl = np.clip(xc / t[..., None], 0, 1)
    inter["reflectance"] = refl
    inter["relit"] = relit

    # 5. MSRCR
    cur = relit
    if "msrcr" in comps and p["msr_weight"] > 0:
        ms = match_brightness(msrcr(xc, p["msr_sigmas"]), relit)
        inter["msrcr"] = ms
        cur = (1 - p["msr_weight"]) * relit + p["msr_weight"] * ms
    inter["retinex"] = cur

    # 6-7. CLAHE and denoising. Default order (DENOISE_AFTER_CLAHE): CLAHE first,
    # then guided-filter denoising whose eps uses the noise actually measured
    # after CLAHE (CLAHE's local histogram stretching re-amplifies residual noise).
    if DENOISE_AFTER_CLAHE:
        if "clahe" in comps:
            cur = clahe_lab(cur, p["clahe_clip"], p["clahe_tiles"], p["clahe_blend"])
        inter["clahe"] = cur
        if "denoise" in comps:
            q = dict(p)
            sig = max(noise_sigma(luminance(cur)), p["sigma_post"] * 0.5)
            q["dn_strength"] = float(np.clip(max(p["dn_strength"], sig / 0.04), 0, 1))
            q["dn_eps"] = max((2.0 * sig) ** 2, 1e-6)
            q["dn_chroma_eps"] = max((4.0 * sig) ** 2, 1e-6)
            cur = denoise(cur, t, q)
        inter["denoised"] = cur
    else:
        if "denoise" in comps:
            cur = denoise(cur, t, p)
        inter["denoised"] = cur
        if "clahe" in comps:
            cur = clahe_lab(cur, p["clahe_clip"], p["clahe_tiles"], p["clahe_blend"])
        inter["clahe"] = cur

    # 8. highlight protection
    if "highlight" in comps:
        cur = protect_highlights(cur, xc, p)
    cur = np.clip(cur, 0, 1).astype(np.float32)
    inter["final"] = cur
    return (cur, inter) if return_intermediates else cur


def clahe_baseline(x: np.ndarray) -> np.ndarray:
    """Plain CLAHE baseline with textbook fixed parameters (clip 2.0, 8x8 tiles)."""
    return clahe_lab(x, 2.0, 8, 1.0)

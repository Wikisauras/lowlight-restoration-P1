"""
analysis.py - Image-condition analysis and adaptive parameter selection.

Every enhancement parameter used in classical.py, deep_learning.py and hybrid.py
is derived here from measured statistics of the input image. Nothing downstream
uses a hard-coded enhancement constant except the FIXED_PARAMS dictionary, which
exists only for the "fixed vs adaptive" ablation.

Measured statistics
-------------------
brightness           mean / median / p95 of luminance Y (BT.601)
entropy              Shannon entropy of the 256-bin luminance histogram (bits)
noise_sigma          Immerkaer (1996) fast noise estimate on Y, edges masked out
noise_rel            noise_sigma / mean brightness (noise relative to signal)
nonuniformity        coefficient of variation of a heavily blurred luminance map
colour_cast          distance of the mean (a*, b*) chroma from neutral (CIELAB)
dark_frac / bright_frac / clipped_frac / black

All images inside the pipeline are float32 RGB in [0, 1], shape (H, W, 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

EPS = 1e-6

# ----------------------------------------------------------------------------
# Input handling / edge-case guards
# ----------------------------------------------------------------------------
MAX_SIDE = 2048     # images larger than this are processed at reduced size
MIN_SIDE = 32       # images smaller than this are upscaled before processing


def to_float_rgb(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert any input (uint8/uint16/float, gray/RGB/RGBA) to float32 RGB [0,1].

    Returns (image, was_grayscale).
    """
    if img is None:
        raise ValueError("Input image is None")
    img = np.asarray(img)
    if img.dtype == np.uint8:
        x = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        x = img.astype(np.float32) / 65535.0
    else:
        x = img.astype(np.float32)
        if x.max() > 1.5:          # float image stored in 0..255
            x = x / 255.0
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    was_gray = False
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
        was_gray = True
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
        was_gray = True
    elif x.ndim == 3 and x.shape[2] == 4:
        x = x[..., :3]
    elif x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Unsupported image shape {img.shape}")
    if not was_gray:
        # an RGB file whose three channels are identical is also grayscale
        was_gray = bool(np.allclose(x[..., 0], x[..., 1], atol=1e-4)
                        and np.allclose(x[..., 1], x[..., 2], atol=1e-4))
    return np.clip(x, 0.0, 1.0), was_gray


def size_guard(x: np.ndarray, max_side: int = MAX_SIDE, min_side: int = MIN_SIDE):
    """Resize huge / tiny images to a safe working size. Returns (img, original_hw)."""
    h, w = x.shape[:2]
    hw = (h, w)
    s = 1.0
    if max(h, w) > max_side:
        s = max_side / max(h, w)
    elif min(h, w) < min_side:
        s = min_side / min(h, w)
    if s != 1.0:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        x = cv2.resize(x, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
        x = np.clip(x, 0, 1)
    return x, hw


def restore_size(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if x.shape[:2] == (h, w):
        return x
    interp = cv2.INTER_CUBIC if h * w > x.shape[0] * x.shape[1] else cv2.INTER_AREA
    if x.ndim == 2:
        return np.clip(cv2.resize(x, (w, h), interpolation=interp), 0, 1)
    return np.clip(cv2.resize(x, (w, h), interpolation=interp), 0, 1)


# ----------------------------------------------------------------------------
# Basic measurements
# ----------------------------------------------------------------------------
def luminance(x: np.ndarray) -> np.ndarray:
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


def entropy(y: np.ndarray) -> float:
    hist = np.histogram(np.clip(y, 0, 1), bins=256, range=(0, 1))[0].astype(np.float64)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def noise_sigma(y: np.ndarray) -> float:
    """Immerkaer fast noise variance estimation, with strong edges excluded.

    J. Immerkaer, "Fast Noise Variance Estimation", CVIU 1996.
    Returns sigma in the [0,1] intensity scale.
    """
    y = y.astype(np.float32)
    h, w = y.shape
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(y, -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    g = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(g, 90)            # drop the 10% strongest edges
    m = g <= thr
    if m.sum() < 16:
        m = np.ones_like(g, bool)
    sigma = math.sqrt(math.pi / 2.0) / 6.0 * float(np.abs(r[m]).mean())
    return sigma


def illumination_map_fast(x: np.ndarray, size: int = 32) -> np.ndarray:
    """Very smooth illumination estimate used for the non-uniformity statistic."""
    m = x.max(axis=2)
    small = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), size / 8)


def colour_cast(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (cast magnitude in a*b* units, per-channel means of valid pixels)."""
    y = luminance(x)
    valid = (y > 0.02) & (x.max(axis=2) < 0.98)
    if valid.sum() < 64:
        valid = np.ones_like(y, bool)
    means = np.array([x[..., c][valid].mean() for c in range(3)], np.float32)
    # normalise so the brightest channel mean is 0.5, then measure chroma in LAB.
    # This makes the cast measure independent of overall exposure.
    rgb = (means / max(means.max(), EPS) * 0.5).reshape(1, 1, 3).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]
    return float(math.hypot(lab[1], lab[2])), means


@dataclass
class ImageStats:
    height: int
    width: int
    was_gray: bool
    brightness: float
    median: float
    p95: float
    entropy: float
    noise_sigma: float
    noise_rel: float
    nonuniformity: float
    colour_cast: float
    channel_means: list = field(default_factory=list)
    dark_frac: float = 0.0
    bright_frac: float = 0.0
    clipped_frac: float = 0.0
    black: bool = False
    condition: str = ""
    flags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def analyse(x: np.ndarray, was_gray: bool = False) -> ImageStats:
    y = luminance(x)
    b = float(y.mean())
    sig = noise_sigma(y)
    T = illumination_map_fast(x)
    nonu = float(T.std() / (T.mean() + 1e-3))
    cast, means = colour_cast(x)
    if was_gray:
        cast = 0.0
    dark = float((y < 0.1).mean())
    bright = float((y > 0.8).mean())
    clipped = float((x.max(axis=2) > 0.98).mean())
    black = bool(np.percentile(y, 99.5) < 3.0 / 255.0)

    flags = []
    if black:
        cond = "black"
    elif b < 0.06:
        cond = "extreme_dark"
    elif b < 0.25:
        cond = "dark"
    elif b > 0.6:
        cond = "bright"
    else:
        cond = "normal"
    if nonu > 0.45 or (dark > 0.3 and bright > 0.02):
        flags.append("non_uniform")
    if sig / max(b, 0.02) > 0.08 or sig > 0.012:
        flags.append("noisy")
    if cast > 6.0:
        flags.append("colour_cast")
    if clipped > 0.005:
        flags.append("saturated_highlights")
    if was_gray:
        flags.append("grayscale")
    return ImageStats(
        height=x.shape[0], width=x.shape[1], was_gray=was_gray,
        brightness=b, median=float(np.median(y)), p95=float(np.percentile(y, 95)),
        entropy=entropy(y), noise_sigma=sig, noise_rel=sig / max(b, 0.01),
        nonuniformity=nonu, colour_cast=cast, channel_means=means.tolist(),
        dark_frac=dark, bright_frac=bright, clipped_frac=clipped, black=black,
        condition=cond, flags=flags,
    )


# ----------------------------------------------------------------------------
# Adaptive parameter selection
# ----------------------------------------------------------------------------
TARGET_EXPOSURE = 0.45   # target mean of the relit illumination (mid-grey-ish)


def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def select_params(s: ImageStats, x: np.ndarray | None = None) -> dict:
    """Map the measured statistics to every pipeline parameter.

    The comments next to each rule are the justification used in the report.
    """
    max_side = max(s.height, s.width)
    min_side = min(s.height, s.width)
    scale = max_side / 512.0

    # --- 1. Illumination / LIME gamma -------------------------------------
    # Relit output = I / T^g  ~  R * T^(1-g).  Choosing g so that the mean
    # illumination t is mapped to TARGET_EXPOSURE gives t^(1-g) = target
    #   => g = 1 - ln(target) / ln(t).
    if x is not None:
        t_mean = float(np.clip(x.max(axis=2).mean(), 0.003, 0.99))
    else:
        t_mean = _clip(s.brightness * 1.3, 0.003, 0.99)
    if t_mean >= TARGET_EXPOSURE:
        gamma = 0.15                                   # already bright: barely relight
    else:
        gamma = 1.0 - math.log(TARGET_EXPOSURE) / math.log(t_mean)
    gamma = _clip(gamma, 0.15, 0.92)

    # --- 2. Guided filter used to refine the illumination ------------------
    gf_radius = int(_clip(round(15 * scale), 2, max(2, min_side // 8)))
    gf_eps = 1e-3 if s.nonuniformity < 0.45 else 4e-3   # smoother T when non-uniform

    # --- 3. Expected noise after enhancement -------------------------------
    # brightness gain of the relighting step ~ t^(1-g)/t = t^(-g)
    gain = _clip(t_mean ** (-gamma), 1.0, 60.0)
    # 8-bit quantisation noise (q^2/12, q = 1/255) is a floor: in an extremely dark
    # frame the signal occupies 1-3 code values and the measured sigma collapses,
    # yet the quantisation steps are amplified by the same gain.
    sigma_eff = math.sqrt(s.noise_sigma ** 2 + (1.0 / 255.0) ** 2 / 12.0)
    sigma_post = _clip(sigma_eff * gain, 0.0, 0.25)

    # --- 4. Retinex (MSRCR) ------------------------------------------------
    base_sig = (15.0, 80.0, 250.0)                       # Jobson et al. scales at ~512px
    msr_sigmas = tuple(max(1.0, b * scale) for b in base_sig)
    # MSRCR helps most for non-uniform illumination but amplifies noise.
    msr_weight = _clip(0.15 + 0.5 * min(s.nonuniformity, 1.0) - 3.0 * sigma_post, 0.0, 0.5)

    # --- 5. Denoising -------------------------------------------------------
    # guided-filter eps scales with expected post-enhancement noise variance
    dn_strength = _clip(sigma_post / 0.04, 0.0, 1.0)      # 0 = off, 1 = strong
    dn_radius = int(_clip(round((2 + 6 * dn_strength) * max(scale, 0.5)), 1, max(1, min_side // 10)))
    dn_eps = max((2.0 * sigma_post) ** 2, 1e-6)
    dn_chroma_eps = max((4.0 * sigma_post) ** 2, 1e-6)    # chroma noise is smoothed harder
    dn_dark_power = _clip(1.0 + 2.0 * dn_strength, 1.0, 3.0)  # how strongly dark areas get more

    # --- 6. CLAHE ----------------------------------------------------------
    # low entropy (flat histogram use) -> more contrast; high noise -> less.
    clahe_clip = _clip(1.2 + 2.6 * (1 - s.entropy / 8.0) - 20.0 * sigma_post, 1.0, 3.5)
    tiles = int(_clip(round(min_side / 96), 2, 16))
    if "non_uniform" in s.flags:
        tiles = int(_clip(tiles * 1.5, 2, 16))
    clahe_blend = _clip(0.5 + 0.5 * (1 - s.entropy / 8.0) * 2, 0.3, 1.0)

    # --- 7. Colour correction ---------------------------------------------
    cc_strength = 0.0 if s.was_gray else _clip((s.colour_cast - 2.0) / 10.0, 0.0, 0.9)

    # --- 8. Highlight protection ------------------------------------------
    # The more of the frame is already bright, the earlier the soft knee starts.
    knee = _clip(0.85 - 0.6 * s.bright_frac, 0.65, 0.88)
    protect_lo = _clip(0.45 + 0.2 * (1 - s.bright_frac), 0.45, 0.7)  # lum where protection starts

    # --- 9. Fusion-weight exponents (hybrid) -------------------------------
    w_exposure = 1.0 + (0.5 if "non_uniform" in s.flags else 0.0)
    w_contrast = 1.0 if s.entropy < 7.0 else 0.6
    w_saturation = 0.0 if s.was_gray else (0.5 if s.colour_cast > 6 else 1.0)
    w_noise = _clip(0.5 + 2.0 * dn_strength, 0.5, 2.5)
    fusion_sigma = 0.2                                    # width of the well-exposedness Gaussian

    # --- 10. Deep learning -------------------------------------------------
    dce_scale = int(_clip(round(max_side / 256), 1, 12))
    dce_strengths = (0.6, 0.8, 1.0, 1.2, 1.4) if s.brightness < 0.15 else (0.4, 0.6, 0.8, 1.0)
    prior_beta = _clip(0.5 + 1.0 * min(s.nonuniformity, 1.0), 0.5, 1.5)

    return dict(
        adaptive=True,
        t_mean=t_mean, gamma=gamma, gain=gain, sigma_post=sigma_post,
        gf_radius=gf_radius, gf_eps=gf_eps,
        msr_sigmas=msr_sigmas, msr_weight=msr_weight,
        dn_strength=dn_strength, dn_radius=dn_radius, dn_eps=dn_eps,
        dn_chroma_eps=dn_chroma_eps, dn_dark_power=dn_dark_power,
        clahe_clip=clahe_clip, clahe_tiles=tiles, clahe_blend=clahe_blend,
        cc_strength=cc_strength, knee=knee, protect_lo=protect_lo,
        w_exposure=w_exposure, w_contrast=w_contrast, w_saturation=w_saturation,
        w_noise=w_noise, fusion_sigma=fusion_sigma,
        dce_scale=dce_scale, dce_strengths=dce_strengths, prior_beta=prior_beta,
    )


# Fixed parameter set (typical literature defaults) - ONLY for the ablation study.
FIXED_PARAMS = dict(
    adaptive=False,
    t_mean=None, gamma=0.8, gain=None, sigma_post=0.02,
    gf_radius=15, gf_eps=1e-3,
    msr_sigmas=(15.0, 80.0, 250.0), msr_weight=0.3,
    dn_strength=0.5, dn_radius=5, dn_eps=(0.03) ** 2, dn_chroma_eps=(0.06) ** 2,
    dn_dark_power=2.0,
    clahe_clip=2.0, clahe_tiles=8, clahe_blend=1.0,
    cc_strength=0.5, knee=0.8, protect_lo=0.6,
    w_exposure=1.0, w_contrast=1.0, w_saturation=1.0, w_noise=1.0, fusion_sigma=0.2,
    dce_scale=4, dce_strengths=(1.0,), prior_beta=1.5,
)


def fixed_params_for(s: ImageStats) -> dict:
    p = dict(FIXED_PARAMS)
    p["dce_scale"] = int(_clip(round(max(s.height, s.width) / 256), 1, 12))  # size-only, not content
    p["gf_radius"] = int(min(p["gf_radius"], max(2, min(s.height, s.width) // 8)))
    p["dn_radius"] = int(min(p["dn_radius"], max(1, min(s.height, s.width) // 10)))
    return p


def describe(s: ImageStats, p: dict) -> list[str]:
    """Human-readable condition report (used by the app)."""
    lines = [
        f"Condition: **{s.condition.replace('_', ' ')}**" + (f" ({', '.join(s.flags)})" if s.flags else ""),
        f"Mean brightness {s.brightness:.3f}, entropy {s.entropy:.2f} bits",
        f"Noise sigma {s.noise_sigma * 255:.2f}/255 (relative {s.noise_rel:.3f}); expected after gain x{p.get('gain') or 0:.1f}: {p['sigma_post'] * 255:.1f}/255",
        f"Illumination non-uniformity (CoV) {s.nonuniformity:.2f}; dark pixels {s.dark_frac * 100:.1f}%, bright {s.bright_frac * 100:.1f}%",
        f"Colour cast {s.colour_cast:.1f} (a*b* units) -> correction strength {p['cc_strength']:.2f}",
        f"Chosen: gamma={p['gamma']:.2f}, CLAHE clip={p['clahe_clip']:.2f} tiles={p['clahe_tiles']}, "
        f"denoise strength={p['dn_strength']:.2f}, MSRCR weight={p['msr_weight']:.2f}, knee={p['knee']:.2f}",
    ]
    return lines

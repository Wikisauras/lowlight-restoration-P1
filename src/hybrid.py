"""
hybrid.py - Hybrid classical + deep-learning restoration, and the top-level
`enhance()` entry point used by the app, evaluation and CLI.

Hybrid pipeline
---------------
1. Classical illumination map T (max-RGB + guided filter) is computed once.
2. ILLUMINATION PRIOR -> DL: the Zero-DCE++ curve map A is modulated per pixel
       A'(p) = A(p) * m(p),   m = (1 - T)^beta / mean((1 - T)^beta)
   so dark regions receive stronger curves and already-lit regions weaker ones;
   m is forced towards 0 where T is close to 1 (never brighten lit regions).
3. Quality-aware strength selection on the prior-guided curves (deep_learning.py).
4. Candidates: classical output, prior-guided DL output, and the (colour-corrected)
   input itself (which wins only where it is already well exposed).
5. Per-pixel fusion weights per candidate k:
       W_k = E_k^we * C_k^wc * S_k^ws * N_k^wn
     E well-exposedness  exp(-(Y-0.5)^2 / 2 sigma^2)       (Mertens et al. 2007)
     C local contrast    |Laplacian(Y)| (smoothed)
     S saturation        std over R,G,B
     N local noise       1 / (1 + local_noise / sigma_post)
   Exponents (we, wc, ws, wn) come from analysis.select_params.
6. Laplacian-pyramid blending of the candidates with Gaussian-pyramid weights
   (avoids seams and halos of naive per-pixel blending).
7. Dark-region denoising of the fused result (strength ~ (1-T)^p * sigma_post).
8. No-blow-out guard: in regions that were already lit, output luminance may not
   exceed the input by more than a small margin; then a soft knee.

Auto mode: analyses the image and runs classical, DL and hybrid; picks the
output with the best no-reference score, with a small prior in favour of the
hybrid (it is the most robust on average) - "select OR combine".
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import analysis as A
from .analysis import luminance, EPS
from .classical import (enhance_classical, guided_filter, denoise, soft_knee, gray_world,
                        estimate_illumination, ALL_COMPONENTS)
from .deep_learning import enhance_dl, fast_quality_score, predict_curves, apply_curves, select_strength

HYBRID_COMPONENTS = ("prior", "quality_aware", "dl_denoise", "w_exposure", "w_contrast", "w_saturation", "w_noise",
                     "pyramid", "dark_denoise", "guard")
# "input_candidate" (use the input itself as a third fusion candidate) is available
# but off by default: in our tests the blow-out guard protects lit regions better.


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def _relight_gamma(t, target=0.45):
    """LIME gamma that maps illumination t to the target exposure (see analysis.py)."""
    t = np.clip(t, 0.003, 0.99)
    return np.clip(1.0 - np.log(target) / np.log(t), 0.05, 0.95)


def illumination_prior(t: np.ndarray, beta: float) -> np.ndarray:
    """Per-pixel curve-strength modulation from the classical illumination map.

    The fine max-RGB map still contains reflectance (a dark object is not a dark
    light), so the prior uses a LOW-FREQUENCY version T_s of T (Gaussian, sigma =
    1/16 of the image size) that captures only the lighting layout.

    Design (chosen after testing a symmetric boost/attenuate prior, which reduced
    PSNR on every condition because Zero-DCE++ already saturates its curves in
    dark regions): the prior only ATTENUATES the learned curves where the scene
    is already lit, relative to the image's own illumination level:
        m = 1 - 0.85 * clip((T_s - tau) / 0.35, 0, 1)^(1/beta),
        tau = max(mean(T_s) + std(T_s), 0.35)
    Uniformly dark scenes get m = 1 (pure Zero-DCE++); spotlights, lamps and
    windows get m -> 0.15 so they are not blown out.
    """
    h, w = t.shape
    s = max(2.0, max(h, w) / 16.0)
    ts = cv2.GaussianBlur(t.astype(np.float32), (0, 0), s)
    tau = max(float(ts.mean() + ts.std()), 0.35)
    lit = np.clip((ts - tau) / 0.35, 0, 1) ** (1.0 / max(beta, 0.1))
    m = 1.0 - 0.85 * lit
    return m.astype(np.float32)


def local_noise_map(y: np.ndarray, r: int = 3) -> np.ndarray:
    """Local std of the high-pass residual (y - guided-smoothed y)."""
    sm = guided_filter(y, y, r, 1e-3)
    res = y - sm
    var = cv2.boxFilter(res * res, -1, (2 * r + 1, 2 * r + 1))
    return np.sqrt(np.maximum(var, 0))


def fusion_weights(cands, p: dict, comps: set, sigma_ref: float):
    ws = []
    for c in cands:
        y = luminance(c)
        w = np.ones_like(y)
        if "w_exposure" in comps:
            e = np.exp(-((c - 0.5) ** 2) / (2 * p["fusion_sigma"] ** 2)).prod(axis=2)
            w *= np.power(e + 1e-6, p["w_exposure"])
        if "w_contrast" in comps:
            # contrast of the noise-free structure: Laplacian of the guided-filtered
            # luminance, otherwise sensor noise would be rewarded as "contrast"
            ys = guided_filter(y, y, 2, max((2 * sigma_ref) ** 2, 1e-5))
            lap = np.abs(cv2.Laplacian(ys, cv2.CV_32F, ksize=3))
            lap = cv2.GaussianBlur(lap, (0, 0), 2)
            w *= np.power(lap + 1e-3, p["w_contrast"])
        if "w_saturation" in comps and p["w_saturation"] > 0:
            s = c.std(axis=2)
            w *= np.power(s + 1e-3, p["w_saturation"])
        if "w_noise" in comps:
            n = local_noise_map(y)
            w *= np.power(1.0 / (1.0 + n / max(sigma_ref, 2e-3)), p["w_noise"])
        ws.append(w.astype(np.float32))
    W = np.stack(ws, 0)
    W = W / (W.sum(0, keepdims=True) + 1e-12)
    return W


def _levels(h, w):
    return int(max(1, min(6, np.floor(np.log2(max(1, min(h, w)) / 8)))))


def pyramid_fuse(cands, W):
    """Laplacian-pyramid fusion (Burt & Adelson) with Gaussian-pyramid weights."""
    h, w = cands[0].shape[:2]
    L = _levels(h, w)
    out_pyr = None
    for c, wk in zip(cands, W):
        gp_w = [wk]
        for _ in range(L):
            gp_w.append(cv2.pyrDown(gp_w[-1]))
        gp = [c]
        for _ in range(L):
            gp.append(cv2.pyrDown(gp[-1]))
        lp = []
        for i in range(L):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[L])
        contrib = [l * g[..., None] for l, g in zip(lp, gp_w)]
        out_pyr = contrib if out_pyr is None else [a + b for a, b in zip(out_pyr, contrib)]
    out = out_pyr[-1]
    for i in range(L - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(out_pyr[i].shape[1], out_pyr[i].shape[0])) + out_pyr[i]
    return np.clip(out, 0, 1)


def denoise_like(y, x_in, t, p):
    """Denoise an enhanced image with strength set from the measured gain.

    expected noise = input noise * local gain (y/x), so the guided-filter eps follows
    the actual amplification of this candidate rather than the classical one.
    """
    yl, xl = luminance(y), luminance(x_in)
    gain = float(np.clip(np.median(yl / np.maximum(xl, 0.01)), 1.0, 60.0))
    sig = A.noise_sigma(xl) * gain
    q = dict(p)
    q["dn_strength"] = float(np.clip(sig / 0.04, 0, 1))
    q["dn_eps"] = max((2.0 * sig) ** 2, 1e-6)
    q["dn_chroma_eps"] = max((4.0 * sig) ** 2, 1e-6)
    q["dn_radius"] = int(np.clip(round(2 + 6 * q["dn_strength"]), 1, max(1, min(y.shape[:2]) // 10)))
    return denoise(y, t, q)


def blowout_guard(out, x_in, t, p):
    """Never blow out regions that were already lit."""
    y_in, y_out = luminance(x_in), luminance(out)
    lit = np.clip((t - 0.55) / 0.35, 0, 1)
    lit = cv2.GaussianBlur(lit, (0, 0), 3)
    limit = np.maximum(y_in * 1.08 + 0.03, y_in)            # small margin above input
    y_cap = np.minimum(y_out, limit)
    # lit regions: capped relative to the input (light sources keep their value);
    # everything else: soft knee against clipping
    y_new = lit * y_cap + (1 - lit) * soft_knee(y_out, p["knee"])
    return np.clip(out * (y_new / np.maximum(y_out, EPS))[..., None], 0, 1)


# ----------------------------------------------------------------------------
# Hybrid
# ----------------------------------------------------------------------------
def enhance_hybrid(x, p, stats=None, components=HYBRID_COMPONENTS, classical_components=ALL_COMPONENTS,
                   scorer="fast", device="auto", model=None, classical_result=None):
    comps = set(components)
    inter = {}
    t0 = time.perf_counter()
    if classical_result is None:
        cls_out, cls_inter = enhance_classical(x, p, classical_components)
    else:
        cls_out, cls_inter = classical_result
    inter.update({f"classical_{k}": v for k, v in cls_inter.items()})
    xc = cls_inter["colour_corrected"]
    t = cls_inter["illumination"]
    t_cls = time.perf_counter() - t0

    # DL with illumination prior
    mod = illumination_prior(t, p["prior_beta"]) if "prior" in comps else None
    inter["prior_modulation"] = mod if mod is not None else np.ones_like(t)
    dl_out, dl_inter = enhance_dl(xc, p, quality_aware="quality_aware" in comps, scorer=scorer,
                                  modulation=mod, device=device, model=model)
    inter["dl_prior_output"] = dl_out
    # classical structure-preserving denoising of the DL candidate (Zero-DCE++ has no
    # noise model and amplifies sensor noise together with the signal)
    if "dl_denoise" in comps:
        dl_out = denoise_like(dl_out, xc, t, p)
    inter["dl_denoised"] = dl_out
    inter["dl_curve_map"] = dl_inter["curve_map"]
    inter["dl_strength"] = dl_inter["strength"]
    inter["dl_strength_table"] = dl_inter["strength_table"]

    cands = [cls_out, dl_out]
    names = ["classical", "dl_prior"]
    if "input_candidate" in comps:
        cands.append(xc)
        names.append("input")
    sigma_ref = max(p["sigma_post"], 2e-3)
    W = fusion_weights(cands, p, comps, sigma_ref)
    inter["fusion_weights"] = {n: W[i] for i, n in enumerate(names)}
    if "pyramid" in comps:
        fused = pyramid_fuse(cands, W)
    else:
        fused = np.clip(sum(c * w[..., None] for c, w in zip(cands, W)), 0, 1)
    inter["fused"] = fused

    out = fused
    if "dark_denoise" in comps:
        sig_now = A.noise_sigma(luminance(out))
        q = dict(p)
        q["dn_strength"] = float(np.clip(sig_now / 0.03, 0, 1))
        q["dn_eps"] = max((1.5 * sig_now) ** 2, 1e-6)
        q["dn_chroma_eps"] = max((3.0 * sig_now) ** 2, 1e-6)
        out = denoise(out, t, q)
    inter["dark_denoised"] = out
    if "guard" in comps:
        out = blowout_guard(out, xc, t, p)
    out = np.clip(out, 0, 1).astype(np.float32)
    inter["final"] = out
    inter["time_classical_s"] = t_cls
    return out, inter


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------
MODES = ("classical", "dl", "hybrid", "auto")


def prepare(img):
    """Edge-case guards + analysis + parameter selection."""
    x, was_gray = A.to_float_rgb(img)
    x, orig_hw = A.size_guard(x)
    stats = A.analyse(x, was_gray)
    params = A.select_params(stats, x)
    return x, was_gray, orig_hw, stats, params


def finalize(y, was_gray, orig_hw):
    y = A.restore_size(y, orig_hw)
    if was_gray:
        g = luminance(y)
        y = np.repeat(g[..., None], 3, axis=2)
    return np.clip(y, 0, 1).astype(np.float32)


def enhance(img, mode="auto", adaptive=True, device="auto", scorer="fast", model=None,
            classical_components=ALL_COMPONENTS, hybrid_components=HYBRID_COMPONENTS):
    """Enhance one image. Returns (output float RGB [0,1], info dict)."""
    t_start = time.perf_counter()
    x, was_gray, orig_hw, stats, params = prepare(img)
    if not adaptive:
        params = A.fixed_params_for(stats)
    info = dict(stats=stats, params=params, mode=mode, intermediates={})

    # Completely black frame: there is no signal to recover - do not invent one.
    if stats.black:
        y = x.copy()
        info["note"] = "Input is (almost) completely black: no recoverable signal, returned unchanged."
        info["chosen"] = "passthrough"
        info["latency_ms"] = (time.perf_counter() - t_start) * 1000
        return finalize(y, was_gray, orig_hw), info

    if mode == "classical":
        y, inter = enhance_classical(x, params, classical_components)
        info["chosen"] = "classical"
    elif mode == "dl":
        y, inter = enhance_dl(x, params, quality_aware=adaptive, scorer=scorer, device=device, model=model)
        info["chosen"] = "dl"
    elif mode == "hybrid":
        y, inter = enhance_hybrid(x, params, stats, hybrid_components, classical_components,
                                  scorer=scorer, device=device, model=model)
        info["chosen"] = "hybrid"
    elif mode == "auto":
        cls = enhance_classical(x, params, classical_components)
        hy_out, inter = enhance_hybrid(x, params, stats, hybrid_components, classical_components,
                                       scorer=scorer, device=device, model=model, classical_result=cls)
        dl_out = inter["dl_prior_output"]
        # plain DL (no prior) is also a candidate for selection
        dl_plain = apply_curves(cls[1]["colour_corrected"], inter["dl_curve_map"], 1.0)
        cands = {"hybrid": hy_out, "classical": cls[0], "dl": dl_plain}
        scores = {k: fast_quality_score(v, x) for k, v in cands.items()}
        scores["hybrid"] += 0.05        # robustness prior
        best = max(scores, key=scores.get)
        y = cands[best]
        info["auto_scores"] = scores
        info["chosen"] = best
    else:
        raise ValueError(f"mode must be one of {MODES}")
    info["intermediates"] = inter
    info["latency_ms"] = (time.perf_counter() - t_start) * 1000
    return finalize(y, was_gray, orig_hw), info

"""
deep_learning.py - Zero-DCE++ low-light enhancement with OFFICIAL pretrained weights,
plus an image-quality-aware curve-strength selection and an optional
illumination prior (used by hybrid.py).

Model: Zero-DCE++ (C. Li, C. Guo, C. C. Loy, "Learning to Enhance Low-Light Image
via Zero-Reference Deep Curve Estimation", TPAMI 2021). The network predicts a
per-pixel, per-channel curve parameter map A in [-1, 1]; the image is enhanced by
iterating the quadratic light-enhancement curve 8 times:

        LE_{n+1}(x) = LE_n(x) + A(x) * (LE_n(x)^2 - LE_n(x))

Architecture reproduced from the official repository
https://github.com/Li-Chongyi/Zero-DCE_extension (Zero-DCE++/model.py); the layer
names match so the official checkpoint `Epoch99.pth` loads with strict=True.

Quality-aware strategy (requirement B4)
--------------------------------------
Because A is predicted once, re-applying the curve with a scaled map k*A is almost
free. We evaluate several strengths k on a down-scaled copy and pick the one with
the best no-reference quality score (fast composite score by default; NIQE or
BRISQUE via pyiqa/piq if requested).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .analysis import luminance, entropy, noise_sigma, EPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "ZeroDCE++_Epoch99.pth")
N_ITER = 8


class CSDN_Tem(nn.Module):
    """Depthwise-separable convolution block used by Zero-DCE++."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEpp(nn.Module):
    def __init__(self, number_f: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def curve_maps(self, x: torch.Tensor, scale_factor: int = 1) -> torch.Tensor:
        """Predict the curve-parameter map A (B,3,H,W) in [-1,1] at full resolution."""
        h, w = x.shape[-2:]
        xd = x if scale_factor == 1 else F.interpolate(x, scale_factor=1 / scale_factor, mode="bilinear",
                                                       align_corners=False)
        x1 = self.relu(self.e_conv1(xd))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        a = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if a.shape[-2:] != (h, w):
            # official model upsamples with nn.UpsamplingBilinear2d (align_corners=True)
            a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=True)
        return a

    @staticmethod
    def apply_curve(x: torch.Tensor, a: torch.Tensor, n_iter: int = N_ITER) -> torch.Tensor:
        for _ in range(n_iter):
            x = x + a * (x * x - x)
        return x

    def forward(self, x, scale_factor: int = 1):
        a = self.curve_maps(x, scale_factor)
        return self.apply_curve(x, a), a


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(weights: str | None = None, device: str = "auto") -> ZeroDCEpp:
    weights = weights or os.environ.get("ZERODCE_WEIGHTS", DEFAULT_WEIGHTS)
    dev = get_device(device)
    key = (os.path.abspath(weights), str(dev))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"Zero-DCE++ weights not found at {weights}. Download Epoch99.pth from "
            "https://github.com/Li-Chongyi/Zero-DCE_extension/tree/main/Zero-DCE++/snapshots_Zero_DCE++ "
            "and place it in weights/ZeroDCE++_Epoch99.pth")
    net = ZeroDCEpp()
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net.eval().to(dev)
    _MODEL_CACHE[key] = net
    return net


def _to_tensor(x: np.ndarray, dev) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().to(dev)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def predict_curves(x: np.ndarray, scale_factor: int = 4, model=None, device="auto") -> np.ndarray:
    """Return curve map A (H,W,3) for image x, handling arbitrary sizes by padding."""
    net = model or load_model(device=device)
    dev = next(net.parameters()).device
    h, w = x.shape[:2]
    sf = int(max(1, scale_factor))
    while sf > 1 and min(h, w) // sf < 8:   # tiny images: estimate at full res
        sf -= 1
    ph, pw = (-h) % sf, (-w) % sf
    xp = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else x
    with torch.no_grad():
        a = net.curve_maps(_to_tensor(xp, dev), sf)
    return _to_numpy((a + 1) / 2)[:h, :w] * 2 - 1      # back to [-1,1]


def apply_curves(x: np.ndarray, a: np.ndarray, strength: float = 1.0, modulation: np.ndarray | None = None,
                 n_iter: int = N_ITER) -> np.ndarray:
    """Apply LE curve with map strength*modulation*A (numpy, fast)."""
    aa = a * strength
    if modulation is not None:
        aa = aa * (modulation[..., None] if modulation.ndim == 2 else modulation)
    aa = np.clip(aa, -1, 1).astype(np.float32)
    y = x.astype(np.float32).copy()
    for _ in range(n_iter):
        y = y + aa * (y * y - y)
    return np.clip(y, 0, 1)


# ----------------------------------------------------------------------------
# No-reference quality scoring (for the quality-aware strategy)
# ----------------------------------------------------------------------------
def fast_quality_score(y: np.ndarray, x_in: np.ndarray | None = None) -> float:
    """Composite no-reference score, higher = better. Designed to be cheap.

    + exposure term     : closeness of mean luminance to 0.5 (Mertens well-exposedness)
    + information term  : luminance entropy / 8
    - clipping penalty  : fraction of pixels saturated in any channel
    - noise penalty     : estimated noise sigma of the output (amplified noise)
    - colour penalty    : gray-world deviation (unnatural casts)
    """
    Y = luminance(y)
    m = float(Y.mean())
    exp_term = float(np.exp(-((m - 0.5) ** 2) / (2 * 0.15 ** 2)))
    info = entropy(Y) / 8.0
    clip = float((y.max(axis=2) > 0.985).mean())
    if x_in is not None:
        clip = max(0.0, clip - float((x_in.max(axis=2) > 0.985).mean()))
    sig = noise_sigma(Y)
    means = y.reshape(-1, 3).mean(0) + EPS
    cast = float(np.abs(means / means.mean() - 1).mean())
    return exp_term + info - 4.0 * clip - 6.0 * sig - 0.5 * cast


_IQA = {}


def iqa_score(y: np.ndarray, metric: str = "niqe") -> float:
    """NIQE/BRISQUE (lower is better) -> returned negated so higher = better."""
    import torch as _t
    if metric not in _IQA:
        import pyiqa
        _IQA[metric] = pyiqa.create_metric(metric, device="cpu")
    t = _t.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)))[None].float()
    return -float(_IQA[metric](t))


def select_strength(x: np.ndarray, a: np.ndarray, strengths, modulation=None, scorer: str = "fast",
                    work_side: int = 384):
    """Quality-aware selection of the curve strength k. Returns (best_k, table)."""
    h, w = x.shape[:2]
    s = min(1.0, work_side / max(h, w))
    if s < 1:
        size = (max(8, int(w * s)), max(8, int(h * s)))
        xs = cv2.resize(x, size, interpolation=cv2.INTER_AREA)
        as_ = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(modulation, size, interpolation=cv2.INTER_AREA) if modulation is not None else None
    else:
        xs, as_, ms = x, a, modulation
    table = []
    for k in strengths:
        y = apply_curves(xs, as_, k, ms)
        if scorer in ("niqe", "brisque"):
            try:
                sc = iqa_score(y, scorer)
            except Exception:
                sc = fast_quality_score(y, xs)
        else:
            sc = fast_quality_score(y, xs)
        table.append((float(k), float(sc)))
    best = max(table, key=lambda t: t[1])[0]
    return best, table


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def enhance_dl(x: np.ndarray, p: dict, quality_aware: bool = True, scorer: str = "fast",
               modulation: np.ndarray | None = None, device: str = "auto", model=None,
               return_intermediates: bool = True):
    """Zero-DCE++ enhancement.

    quality_aware=False reproduces the plain official model (strength 1).
    modulation (H,W) - optional per-pixel strength map (illumination prior from hybrid.py).
    """
    t0 = time.perf_counter()
    a = predict_curves(x, p.get("dce_scale", 4), model=model, device=device)
    t_net = time.perf_counter() - t0
    if quality_aware and len(p.get("dce_strengths", (1.0,))) > 1:
        k, table = select_strength(x, a, p["dce_strengths"], modulation, scorer)
    else:
        k, table = 1.0, [(1.0, float("nan"))]
    y = apply_curves(x, a, k, modulation)
    inter = dict(curve_map=a, strength=k, strength_table=table, net_time_s=t_net)
    return (y, inter) if return_intermediates else y

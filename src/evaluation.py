"""
evaluation.py - Comparative evaluation, ablation and automated failure analysis.

Compares: raw input, CLAHE baseline, classical, deep learning (official Zero-DCE++),
hybrid (and auto mode). Metrics:

  paired (ground truth available)      PSNR, SSIM, LPIPS (AlexNet), CIEDE2000 vs GT,
                                       edge preservation index (EPI) vs GT
  no-reference (any image)             NIQE, BRISQUE
  efficiency                           latency on CPU and on GPU (if CUDA available)
  noise                                noise sigma before / after (Immerkaer), noise
                                       amplification = sigma_out / (sigma_in * gain)
  failure analysis                     % newly clipped pixels, colour shift (dE2000),
                                       halo score, noise amplification; worst cases saved

Usage (laptop, full run):
  python -m src.evaluation --paired LOL=data/LOL/eval15 --unpaired data/unpaired ^
         --synthetic data/synthetic --out results --ablation --device auto
Every input folder is optional; missing folders are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from . import analysis as A
from .analysis import luminance
from .classical import clahe_baseline, ALL_COMPONENTS
from .hybrid import enhance, HYBRID_COMPONENTS
from .deep_learning import load_model, get_device

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
METHODS = ("raw", "clahe", "classical", "dl", "hybrid", "auto")
COLORS = {"raw": "#52514e", "clahe": "#eda100", "classical": "#2a78d6", "dl": "#eb6834",
          "hybrid": "#1baf7a", "auto": "#6250d6"}


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)          # unicode-safe on Windows
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imwrite(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x8 = (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    if x8.ndim == 3:
        x8 = cv2.cvtColor(x8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", x8)
    buf.tofile(path)


def label_tile(t, text, scale=0.5):
    """Draw a label in the top-left corner (cv2 text drawing needs uint8)."""
    u = (np.clip(t, 0, 1) * 255).astype(np.uint8).copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(u, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(u, text, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return u.astype(np.float32) / 255


def list_images(d):
    if not d or not os.path.isdir(d):
        return []
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(EXTS))


def find_pairs(root):
    """root/low + root/high (LOL layout) with matching file names."""
    lows = list_images(os.path.join(root, "low"))
    pairs = []
    for f in lows:
        stem = os.path.splitext(os.path.basename(f))[0]
        cands = [g for g in list_images(os.path.join(root, "high")) if os.path.splitext(os.path.basename(g))[0] == stem]
        if cands:
            pairs.append((f, cands[0]))
    return pairs


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
class IQA:
    """Lazy wrappers around pyiqa / piq. Missing metrics return NaN (reported)."""

    def __init__(self, device="cpu"):
        self.device = device
        self._m = {}
        self.unavailable = {}

    def _get(self, name):
        if name in self._m or name in self.unavailable:
            return self._m.get(name)
        try:
            import pyiqa
            self._m[name] = pyiqa.create_metric(name, device=self.device)
        except Exception as e:  # noqa
            if name == "brisque":
                try:
                    import piq  # noqa
                    self._m[name] = "piq"
                    return self._m[name]
                except Exception:
                    pass
            self.unavailable[name] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"[warn] metric {name} unavailable: {self.unavailable[name]}")
        return self._m.get(name)

    @staticmethod
    def _t(x):
        import torch
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float()

    def nr(self, name, x):
        m = self._get(name)
        if m is None:
            return float("nan")
        import torch
        with torch.no_grad():
            if m == "piq":
                import piq
                return float(piq.brisque(self._t(x), data_range=1.0))
            return float(m(self._t(x).to(self.device)))

    def lpips(self, x, ref):
        m = self._get("lpips")
        if m is None:
            return float("nan")
        import torch
        with torch.no_grad():
            return float(m(self._t(x).to(self.device), self._t(ref).to(self.device)))


def psnr(x, ref):
    mse = float(np.mean((x.astype(np.float64) - ref) ** 2))
    return 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)


def ssim(x, ref):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ref, x, channel_axis=2, data_range=1.0))


def _small(x, side=512):
    s = min(1.0, side / max(x.shape[:2]))
    if s < 1:
        x = cv2.resize(x, (int(x.shape[1] * s), int(x.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return x


def edge_preservation_index(x, ref):
    """EPI (Sattar et al. 1997): correlation of Laplacian high-pass of output and reference."""
    a = cv2.Laplacian(luminance(x), cv2.CV_32F, ksize=3)
    b = cv2.Laplacian(luminance(ref), cv2.CV_32F, ksize=3)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def delta_e(x, ref):
    from skimage.color import rgb2lab, deltaE_ciede2000
    return float(deltaE_ciede2000(rgb2lab(_small(ref)), rgb2lab(_small(x))).mean())


def colour_shift(x, x_in):
    """Hue/chroma shift vs the input, independent of brightness: dE2000 after
    giving the input the output's L* channel."""
    from skimage.color import rgb2lab, deltaE_ciede2000
    lo, li = rgb2lab(_small(x)), rgb2lab(_small(x_in))
    y = luminance(_small(x_in))
    valid = y > 0.02
    li[..., 0] = lo[..., 0]
    # input chroma of a dark pixel is tiny; scale to the output's chroma level
    ci = np.hypot(li[..., 1], li[..., 2]).mean() + 1e-6
    co = np.hypot(lo[..., 1], lo[..., 2]).mean()
    li[..., 1:] *= co / ci
    d = deltaE_ciede2000(li, lo)
    return float(d[valid].mean()) if valid.any() else float(d.mean())


def clipped_pct(x, ref):
    """Percentage of pixels newly saturated (any channel >= 0.99) vs the reference."""
    return float(100 * max(0.0, (x.max(axis=2) >= 0.99).mean() - (ref.max(axis=2) >= 0.99).mean()))


def halo_score(x, ref):
    """Ringing/overshoot energy in a band around strong edges, relative to the
    reference (brightness-normalised). 1.0 = same as reference, >1 = halos."""
    xs, rs = _small(x), _small(ref)
    yo, yr = luminance(xs), luminance(rs)
    yr_n = yr * (yo.mean() / max(yr.mean(), 1e-4))
    e = cv2.Canny((np.clip(cv2.GaussianBlur(yr_n, (0, 0), 1.5), 0, 1) * 255).astype(np.uint8), 40, 120)
    k1, k2 = np.ones((3, 3), np.uint8), np.ones((15, 15), np.uint8)
    band = (cv2.dilate(e, k2) > 0) & ~(cv2.dilate(e, k1) > 0)
    if band.sum() < 50:
        return 1.0
    hp_o = np.abs(yo - cv2.GaussianBlur(yo, (0, 0), 8))
    hp_r = np.abs(yr_n - cv2.GaussianBlur(yr_n, (0, 0), 8))
    return float(hp_o[band].mean() / (hp_r[band].mean() + 1e-4))


def noise_stats(x, x_in):
    s_in = A.noise_sigma(luminance(x_in))
    s_out = A.noise_sigma(luminance(x))
    gain = max(luminance(x).mean(), 1e-4) / max(luminance(x_in).mean(), 1e-4)
    return s_in, s_out, s_out / max(s_in * gain, 1e-6)


# ----------------------------------------------------------------------------
# Method runners
# ----------------------------------------------------------------------------
def run_method(method, img_u8, device="auto", model=None, adaptive=True,
               classical_components=ALL_COMPONENTS, hybrid_components=HYBRID_COMPONENTS):
    x, _ = A.to_float_rgb(img_u8)
    t = time.perf_counter()
    info = {}
    if method == "raw":
        y = x
    elif method == "clahe":
        y = clahe_baseline(x)
    else:
        y, info = enhance(img_u8, method, adaptive=adaptive, device=device, model=model,
                          classical_components=classical_components, hybrid_components=hybrid_components)
    if device != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    ms = (time.perf_counter() - t) * 1000
    return y, ms, info


def ablation_variants():
    v = {}
    # classical
    v["classical/full"] = ("classical", True, ALL_COMPONENTS, HYBRID_COMPONENTS)
    for c in ALL_COMPONENTS:
        v[f"classical/-{c}"] = ("classical", True, tuple(k for k in ALL_COMPONENTS if k != c), HYBRID_COMPONENTS)
    v["classical/fixed_params"] = ("classical", False, ALL_COMPONENTS, HYBRID_COMPONENTS)
    # deep learning
    v["dl/official_k1"] = ("dl", False, ALL_COMPONENTS, HYBRID_COMPONENTS)
    v["dl/quality_aware"] = ("dl", True, ALL_COMPONENTS, HYBRID_COMPONENTS)
    # hybrid
    v["hybrid/full"] = ("hybrid", True, ALL_COMPONENTS, HYBRID_COMPONENTS)
    for c in HYBRID_COMPONENTS:
        v[f"hybrid/-{c}"] = ("hybrid", True, ALL_COMPONENTS, tuple(k for k in HYBRID_COMPONENTS if k != c))
    v["hybrid/fixed_params"] = ("hybrid", False, ALL_COMPONENTS, HYBRID_COMPONENTS)
    return v


# ----------------------------------------------------------------------------
# Evaluation loops
# ----------------------------------------------------------------------------
def evaluate_set(name, items, out_dir, iqa, methods=METHODS, device="auto", model=None,
                 save_outputs=True, paired=True):
    """items: list of (low_path, gt_path or None). Returns per-image DataFrame."""
    rows = []
    for i, (fl, fh) in enumerate(items):
        low = imread(fl)
        x_in, _ = A.to_float_rgb(low)
        gt = A.to_float_rgb(imread(fh))[0] if fh else None
        if gt is not None and gt.shape != x_in.shape:
            gt = cv2.resize(gt, (x_in.shape[1], x_in.shape[0]), interpolation=cv2.INTER_AREA)
        stem = os.path.splitext(os.path.basename(fl))[0]
        cond = stem.split("__")[0] if "__" in stem else name
        stats = A.analyse(x_in)
        for m in methods:
            y, ms, info = run_method(m, low, device, model)
            r = dict(dataset=name, image=stem, condition=cond, method=m, latency_ms=ms,
                     in_brightness=stats.brightness, in_condition=stats.condition,
                     chosen=info.get("chosen", m))
            s_in, s_out, amp = noise_stats(y, x_in)
            r.update(noise_in=s_in * 255, noise_out=s_out * 255, noise_amp=amp,
                     niqe=iqa.nr("niqe", y), brisque=iqa.nr("brisque", y),
                     colour_shift=colour_shift(y, x_in) if m != "raw" else 0.0)
            ref = gt if gt is not None else x_in
            r.update(clipped_pct=clipped_pct(y, ref), halo=halo_score(y, ref),
                     epi=edge_preservation_index(y, ref))
            if gt is not None:
                r.update(psnr=psnr(y, gt), ssim=ssim(y, gt), lpips=iqa.lpips(y, gt), delta_e=delta_e(y, gt))
            rows.append(r)
            if save_outputs:
                imwrite(os.path.join(out_dir, "outputs", name, m, stem + ".png"), y)
        print(f"  [{name}] {i + 1}/{len(items)} {stem}")
    return pd.DataFrame(rows)


def run_ablation(name, items, iqa, device="auto", model=None, limit=0):
    rows = []
    if limit and len(items) > limit:            # evenly spaced -> covers every condition
        idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
        items = [items[i] for i in idx]
    variants = ablation_variants()
    for i, (fl, fh) in enumerate(items):
        low = imread(fl)
        x_in, _ = A.to_float_rgb(low)
        gt = A.to_float_rgb(imread(fh))[0] if fh else None
        if gt is not None and gt.shape != x_in.shape:
            gt = cv2.resize(gt, (x_in.shape[1], x_in.shape[0]), interpolation=cv2.INTER_AREA)
        stem = os.path.splitext(os.path.basename(fl))[0]
        for vname, (mode, adaptive, cc, hc) in variants.items():
            y, ms, _ = run_method(mode, low, device, model, adaptive, cc, hc)
            s_in, s_out, amp = noise_stats(y, x_in)
            r = dict(dataset=name, image=stem, variant=vname, group=vname.split("/")[0], latency_ms=ms,
                     niqe=iqa.nr("niqe", y), noise_amp=amp, clipped_pct=clipped_pct(y, gt if gt is not None else x_in),
                     halo=halo_score(y, gt if gt is not None else x_in))
            if gt is not None:
                r.update(psnr=psnr(y, gt), ssim=ssim(y, gt), lpips=iqa.lpips(y, gt), delta_e=delta_e(y, gt))
            rows.append(r)
        print(f"  [ablation {name}] {i + 1}/{len(items)}")
    return pd.DataFrame(rows)


def latency_benchmark(items, n=10, repeats=3, model_cpu=None):
    """Median latency per method on CPU and (if available) GPU."""
    import torch
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for dev in devs:
        model = load_model(device=dev)
        for fl, _ in items[:n]:
            low = imread(fl)
            run_method("hybrid", low, dev, model)          # warm-up
            for m in METHODS[1:]:
                ts = [run_method(m, low, dev, model)[1] for _ in range(repeats)]
                rows.append(dict(device=dev, method=m, image=os.path.basename(fl),
                                 h=low.shape[0], w=low.shape[1], latency_ms=float(np.median(ts))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Failure analysis
# ----------------------------------------------------------------------------
FAIL_METRICS = {"clipped_pct": "over-exposure (% newly clipped)", "colour_shift": "colour shift (dE2000)",
                "halo": "halo score", "noise_amp": "noise amplification"}


def failure_analysis(df, out_dir, k=3):
    os.makedirs(os.path.join(out_dir, "failures"), exist_ok=True)
    recs = []
    for m in ("classical", "dl", "hybrid"):
        d = df[df.method == m]
        if d.empty:
            continue
        for met, label in FAIL_METRICS.items():
            worst = d.sort_values(met, ascending=False).head(k)
            for _, r in worst.iterrows():
                recs.append(dict(method=m, metric=met, value=r[met], dataset=r.dataset, image=r.image,
                                 condition=r.condition))
    fdf = pd.DataFrame(recs)
    fdf.to_csv(os.path.join(out_dir, "failures.csv"), index=False)
    # montage of the hybrid worst cases: input | classical | dl | hybrid (| GT)
    for _, r in fdf[fdf.method == "hybrid"].drop_duplicates(["dataset", "image"]).head(8).iterrows():
        tiles, labels = [], []
        base = os.path.join(out_dir, "outputs", r.dataset)
        for m in ("raw", "classical", "dl", "hybrid"):
            p = os.path.join(base, m, r.image + ".png")
            if os.path.isfile(p):
                tiles.append(imread(p).astype(np.float32) / 255)
                labels.append(m if m != "raw" else "input")
        if not tiles:
            continue
        h = 300
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        tiles = [label_tile(t, lab, 0.55) for t, lab in zip(tiles, labels)]
        tiles[-1] = label_tile(tiles[-1], f"hybrid  {r.metric}={r.value:.2f}", 0.55)
        imwrite(os.path.join(out_dir, "failures", f"hybrid_{r.metric}_{r.dataset}_{r.image}.png"), np.concatenate(tiles, 1))
    return fdf


# ----------------------------------------------------------------------------
# Summaries, plots, markdown tables
# ----------------------------------------------------------------------------
AGG = ["psnr", "ssim", "lpips", "delta_e", "niqe", "brisque", "latency_ms", "noise_in", "noise_out",
       "noise_amp", "epi", "clipped_pct", "colour_shift", "halo"]


def summarise(df, by=("dataset", "method")):
    cols = [c for c in AGG if c in df.columns]
    s = df.groupby(list(by))[cols].mean().reset_index()
    order = {m: i for i, m in enumerate(METHODS)}
    if "method" in s:
        s["_o"] = s.method.map(order)
        s = s.sort_values(list(by[:-1]) + ["_o"]).drop(columns="_o")
    return s


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#b0afa8")
    ax.spines["bottom"].set_color("#b0afa8")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_all(summary, abl, lat, df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    # 1. method comparison per dataset
    for ds, d in summary.groupby("dataset"):
        mets = [(m, lab, better) for m, lab, better in
                [("psnr", "PSNR (dB) ↑", 1), ("ssim", "SSIM ↑", 1), ("lpips", "LPIPS ↓", -1),
                 ("niqe", "NIQE ↓", -1), ("brisque", "BRISQUE ↓", -1), ("noise_amp", "Noise amplification ↓", -1)]
                if m in d and d[m].notna().any()]
        fig, axes = plt.subplots(1, len(mets), figsize=(3.1 * len(mets), 3.2))
        axes = np.atleast_1d(axes)
        for ax, (m, lab, _) in zip(axes, mets):
            vals = d[m].values
            bars = ax.bar(d.method, vals, color=[COLORS[k] for k in d.method], width=0.65, edgecolor="white",
                          linewidth=2)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom",
                            fontsize=8, color="#0b0b0b")
            ax.set_title(lab, fontsize=10, color="#0b0b0b")
            ax.tick_params(axis="x", rotation=40)
            _style(ax)
        fig.suptitle(f"Method comparison - {ds}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, f"comparison_{ds}.png"), dpi=150)
        plt.close(fig)
    # 2. per-condition PSNR (synthetic)
    if "condition" in df and "psnr" in df and df.condition.nunique() > 1:
        d = df.dropna(subset=["psnr"])
        d = d[d.dataset == "synthetic"] if (d.dataset == "synthetic").any() else d
        piv = d.groupby(["condition", "method"]).psnr.mean().unstack()
        piv = piv[[m for m in METHODS if m in piv.columns]]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        n = len(piv.columns)
        xs = np.arange(len(piv.index))
        wdt = 0.8 / n
        for j, m in enumerate(piv.columns):
            ax.bar(xs + j * wdt - 0.4 + wdt / 2, piv[m].values, wdt * 0.9, color=COLORS[m], label=m)
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace("_", " ") for c in piv.index])
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Generalisation across illumination conditions (PSNR, synthetic)", fontsize=10)
        ax.legend(frameon=False, ncol=n, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "per_condition_psnr.png"), dpi=150)
        plt.close(fig)
    # 3. ablation deltas
    if abl is not None and not abl.empty:
        key = "psnr" if "psnr" in abl and abl.psnr.notna().any() else "niqe"
        g = abl.groupby("variant")[key].mean()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, grp in zip(axes, ("classical", "hybrid")):
            full = g.get(f"{grp}/full", np.nan)
            vs = g[[v for v in g.index if v.startswith(grp + "/") and v != f"{grp}/full"]]
            delta = (vs - full).sort_values()
            col = ["#e34948" if (v < 0) == (key == "psnr") else "#1baf7a" for v in delta.values]
            ax.barh([v.split("/")[1] for v in delta.index], delta.values, color=col, height=0.6)
            ax.axvline(0, color="#52514e", linewidth=1)
            for yv, v in enumerate(delta.values):
                ax.text(v, yv, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
            ax.set_title(f"{grp}: Δ{key.upper()} vs full ({full:.2f})", fontsize=10)
            _style(ax)
            ax.xaxis.grid(True, color="#e6e5e0")
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "ablation.png"), dpi=150)
        plt.close(fig)
    # 4. latency
    if lat is not None and not lat.empty:
        g = lat.groupby(["device", "method"]).latency_ms.median().unstack(0)
        g = g.reindex([m for m in METHODS[1:] if m in g.index])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        xs = np.arange(len(g.index))
        devs = list(g.columns)
        for j, dv in enumerate(devs):
            ax.bar(xs + (j - (len(devs) - 1) / 2) * 0.38, g[dv].values, 0.36,
                   color="#2a78d6" if dv == "cpu" else "#eb6834", label=dv.upper())
            for xv, v in zip(xs, g[dv].values):
                ax.text(xv + (j - (len(devs) - 1) / 2) * 0.38, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(g.index)
        ax.set_ylabel("median latency (ms / image)")
        ax.set_title("Processing latency", fontsize=10)
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "latency.png"), dpi=150)
        plt.close(fig)
    # 5. noise before / after
    d = df[df.method.isin(["classical", "dl", "hybrid"])]
    if not d.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m in ("dl", "classical", "hybrid"):
            q = d[d.method == m]
            ax.scatter(q.noise_in, q.noise_amp, s=30, color=COLORS[m], label=m, edgecolor="white", linewidth=1)
        ax.axhline(1, color="#52514e", linewidth=1, linestyle="--")
        ax.text(ax.get_xlim()[1], 1, " = noise grows with signal", va="bottom", ha="right", fontsize=8)
        ax.set_xlabel("input noise sigma (/255)")
        ax.set_ylabel("noise amplification (σout / σin·gain)")
        ax.set_title("Noise suppression", fontsize=10)
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "noise.png"), dpi=150)
        plt.close(fig)


def example_grid(out_dir, dataset, images, gt_dir=None, name="examples"):
    """input | clahe | classical | dl | hybrid (| GT) rows for selected images."""
    rows = []
    for stem in images:
        tiles = []
        for m in ("raw", "clahe", "classical", "dl", "hybrid"):
            p = os.path.join(out_dir, "outputs", dataset, m, stem + ".png")
            if os.path.isfile(p):
                tiles.append((m if m != "raw" else "input", imread(p).astype(np.float32) / 255))
        if gt_dir:
            g = [f for f in list_images(gt_dir) if os.path.splitext(os.path.basename(f))[0] == stem]
            if g:
                tiles.append(("ground truth", A.to_float_rgb(imread(g[0]))[0]))
        if not tiles:
            continue
        h = 220
        row = []
        for lab, t in tiles:
            t = cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
            row.append(label_tile(t, lab, 0.45))
        rows.append(np.concatenate(row, 1))
    if rows:
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, wmax - r.shape[1]), (0, 0)), constant_values=1) for r in rows]
        imwrite(os.path.join(out_dir, "plots", f"{name}_{dataset}.png"), np.concatenate(rows, 0))


def fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def markdown_tables(summary, abl, lat, out_dir, iqa):
    L = ["# Auto-generated result tables\n"]
    if iqa.unavailable:
        L.append("Metrics unavailable in this run: " + ", ".join(iqa.unavailable) + "\n")
    for ds, d in summary.groupby("dataset"):
        L.append(f"\n## {ds}\n")
        L.append("| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ΔE00 ↓ | NIQE ↓ | BRISQUE ↓ | Noise σ in→out (/255) | Noise amp ↓ | EPI ↑ | Clipped % ↓ | Halo | Latency ms |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            L.append(f"| {r.method} | {fmt(r.get('psnr'))} | {fmt(r.get('ssim'), 3)} | {fmt(r.get('lpips'), 3)} | "
                     f"{fmt(r.get('delta_e'))} | {fmt(r.get('niqe'))} | {fmt(r.get('brisque'))} | "
                     f"{fmt(r.get('noise_in'))}→{fmt(r.get('noise_out'))} | {fmt(r.get('noise_amp'))} | "
                     f"{fmt(r.get('epi'), 3)} | {fmt(r.get('clipped_pct'))} | {fmt(r.get('halo'))} | {fmt(r.get('latency_ms'), 0)} |")
    if abl is not None and not abl.empty:
        cols = [c for c in ("psnr", "ssim", "lpips", "niqe", "noise_amp", "clipped_pct", "halo") if c in abl and abl[c].notna().any()]
        g = abl.groupby("variant")[cols].mean()
        L.append("\n## Ablation (mean over ablation images)\n")
        L.append("| Variant | " + " | ".join(c.upper() for c in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        order = list(ablation_variants().keys())
        for v in [o for o in order if o in g.index]:
            L.append(f"| {v} | " + " | ".join(fmt(g.loc[v, c], 3) for c in cols) + " |")
    if lat is not None and not lat.empty:
        g = lat.groupby(["method", "device"]).latency_ms.median().unstack()
        L.append("\n## Latency (median ms / image)\n")
        L.append("| Method | " + " | ".join(d.upper() for d in g.columns) + " |")
        L.append("|---|" + "---|" * len(g.columns))
        for m in [x for x in METHODS[1:] if x in g.index]:
            L.append(f"| {m} | " + " | ".join(fmt(g.loc[m, d], 0) for d in g.columns) + " |")
    with open(os.path.join(out_dir, "report_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate low-light enhancement methods")
    ap.add_argument("--paired", action="append", default=[], help="NAME=DIR with DIR/low and DIR/high (repeatable)")
    ap.add_argument("--synthetic", default=None, help="synthetic paired dir (low/, high/)")
    ap.add_argument("--unpaired", default=None, help="dir whose sub-folders are unpaired datasets (DICM, LIME, ...)")
    ap.add_argument("--unpaired_single", action="append", default=[], help="NAME=DIR of one unpaired folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--weights", default=None, help="Zero-DCE++ weights (default: official)")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--ablation_limit", type=int, default=0)
    ap.add_argument("--latency_n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="max images per dataset (debug)")
    ap.add_argument("--no_save", action="store_true")
    a = ap.parse_args(argv)

    if a.weights:
        os.environ["ZERODCE_WEIGHTS"] = a.weights
    os.makedirs(a.out, exist_ok=True)
    dev = str(get_device(a.device))
    model = load_model(a.weights, device=dev)
    iqa = IQA(device=dev)
    print(f"device={dev}  weights={a.weights or 'official Zero-DCE++ Epoch99'}")

    paired_sets = []
    for spec in a.paired:
        n, d = spec.split("=", 1)
        paired_sets.append((n, find_pairs(d), os.path.join(d, "high")))
    if a.synthetic:
        paired_sets.append(("synthetic", find_pairs(a.synthetic), os.path.join(a.synthetic, "high")))
    unpaired_sets = []
    if a.unpaired and os.path.isdir(a.unpaired):
        for sub in sorted(os.listdir(a.unpaired)):
            fs = list_images(os.path.join(a.unpaired, sub))
            if fs:
                unpaired_sets.append((sub, [(f, None) for f in fs]))
    for spec in a.unpaired_single:
        n, d = spec.split("=", 1)
        fs = list_images(d)
        if fs:
            unpaired_sets.append((n, [(f, None) for f in fs]))

    dfs, abls = [], []
    for n, items, _ in paired_sets:
        if not items:
            print(f"[skip] {n}: no pairs found")
            continue
        items = items[: a.limit] if a.limit else items
        print(f"== paired set {n}: {len(items)} pairs")
        dfs.append(evaluate_set(n, items, a.out, iqa, device=dev, model=model, save_outputs=not a.no_save))
        if a.ablation:
            abls.append(run_ablation(n, items, iqa, dev, model, a.ablation_limit))
    for n, items in unpaired_sets:
        items = items[: a.limit] if a.limit else items
        print(f"== unpaired set {n}: {len(items)} images")
        dfs.append(evaluate_set(n, items, a.out, iqa, device=dev, model=model, save_outputs=not a.no_save,
                                paired=False))
    if not dfs:
        print("nothing to evaluate")
        return
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(a.out, "per_image.csv"), index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    if "condition" in df and (df.dataset == "synthetic").any():
        summarise(df[df.dataset == "synthetic"], ("condition", "method")).to_csv(
            os.path.join(a.out, "summary_by_condition.csv"), index=False)
    abl = pd.concat(abls, ignore_index=True) if abls else None
    if abl is not None:
        abl.to_csv(os.path.join(a.out, "ablation_per_image.csv"), index=False)
        cols = [c for c in AGG if c in abl.columns]
        abl.groupby(["dataset", "variant"])[cols].mean().reset_index().to_csv(os.path.join(a.out, "ablation.csv"), index=False)
    all_items = [it for _, items, _ in paired_sets for it in items] + [it for _, items in unpaired_sets for it in items]
    lat = latency_benchmark(all_items, n=a.latency_n) if a.latency_n else None
    if lat is not None:
        lat.to_csv(os.path.join(a.out, "latency.csv"), index=False)
    failure_analysis(df, a.out)
    plot_all(summary, abl, lat, df, a.out)
    for n, items, gdir in paired_sets:
        if items:
            stems = [os.path.splitext(os.path.basename(f))[0] for f, _ in items]
            pick = stems[:: max(1, len(stems) // 5)][:5]
            example_grid(a.out, n, pick, gdir)
    for n, items in unpaired_sets:
        stems = [os.path.splitext(os.path.basename(f))[0] for f, _ in items]
        example_grid(a.out, n, stems[:: max(1, len(stems) // 4)][:4])
    markdown_tables(summary, abl, lat, a.out, iqa)
    with open(os.path.join(a.out, "run_info.json"), "w") as f:
        json.dump(dict(device=dev, weights=a.weights or "official", unavailable_metrics=iqa.unavailable,
                       datasets=[n for n, i, _ in paired_sets if i] + [n for n, _ in unpaired_sets]), f, indent=2)
    print(summary.to_string())
    print(f"\nresults written to {a.out}")


if __name__ == "__main__":
    main()

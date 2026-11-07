"""
train.py - Fine-tune Zero-DCE++ (starting from the OFFICIAL weights) on LOL.

Loss = zero-reference losses of Zero-DCE++ (as in the official training script)
         1600 * L_TV(A)            illumination smoothness of the curve maps
       +    1 * L_spa              spatial consistency (keeps local contrast/edges)
       +    5 * L_col              colour constancy (gray-world)
       +   10 * L_exp(E=0.6)       exposure control
     [ + lambda_pair * (L1 + (1 - SSIM)) against the LOL normal-light image ]  optional

The paired term is our adaptation: LOL provides ground truth, so we use it when
available ("fine-tune wherever suitable training data is available").

Usage (on the laptop GPU):
  python -m src.train --low data/LOL/our485/low --high data/LOL/our485/high --epochs 30
  python -m src.train --low data/LOL/our485/low --epochs 30 --zero_ref_only
Output: weights/ZeroDCE++_LOL_finetuned.pth  (use with --weights in evaluation/app)
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .deep_learning import ZeroDCEpp, DEFAULT_WEIGHTS, ROOT


# ---------------------------------------------------------------- losses ----
class L_color(nn.Module):
    def forward(self, x):
        m = x.mean([2, 3], keepdim=True)
        r, g, b = m[:, 0], m[:, 1], m[:, 2]
        return torch.sqrt((r - g) ** 4 + (r - b) ** 4 + (g - b) ** 4 + 1e-12).mean()


class L_spa(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                          [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, -1, 0]]], dtype=torch.float32).unsqueeze(1)
        self.register_buffer("k", k)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enh):
        o = self.pool(org.mean(1, keepdim=True))
        e = self.pool(enh.mean(1, keepdim=True))
        do = F.conv2d(o, self.k, padding=1)
        de = F.conv2d(e, self.k, padding=1)
        return ((do - de) ** 2).sum(1).mean()


class L_exp(nn.Module):
    def __init__(self, patch=16, E=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch)
        self.E = E

    def forward(self, x):
        return ((self.pool(x.mean(1, keepdim=True)) - self.E) ** 2).mean()


def L_tv(a):
    b, _, h, w = a.shape
    h_tv = ((a[:, :, 1:, :] - a[:, :, :-1, :]) ** 2).sum() / ((h - 1) * w)
    w_tv = ((a[:, :, :, 1:] - a[:, :, :, :-1]) ** 2).sum() / (h * (w - 1))
    return 2 * (h_tv + w_tv) / b


def ssim_loss(x, y):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    sx = F.avg_pool2d(x * x, 7, 1, 3) - mx ** 2
    sy = F.avg_pool2d(y * y, 7, 1, 3) - my ** 2
    sxy = F.avg_pool2d(x * y, 7, 1, 3) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))
    return 1 - s.mean()


# ---------------------------------------------------------------- data ------
class LowLightSet(Dataset):
    def __init__(self, low_dir, high_dir=None, size=256):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.low = sorted(sum([glob.glob(os.path.join(low_dir, e)) for e in exts], []))
        if not self.low:
            raise FileNotFoundError(f"no images in {low_dir}")
        self.high_dir = high_dir
        self.size = size

    def __len__(self):
        return len(self.low)

    def _read(self, f):
        return cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    def __getitem__(self, i):
        lo = self._read(self.low[i])
        hi = None
        if self.high_dir:
            fh = os.path.join(self.high_dir, os.path.basename(self.low[i]))
            hi = self._read(fh) if os.path.isfile(fh) else None
        h, w = lo.shape[:2]
        s = self.size
        if min(h, w) < s:
            lo = cv2.resize(lo, (max(s, w), max(s, h)))
            if hi is not None:
                hi = cv2.resize(hi, (lo.shape[1], lo.shape[0]))
            h, w = lo.shape[:2]
        y0, x0 = random.randint(0, h - s), random.randint(0, w - s)
        lo = lo[y0:y0 + s, x0:x0 + s]
        if hi is not None:
            hi = hi[y0:y0 + s, x0:x0 + s]
        if random.random() < 0.5:
            lo = lo[:, ::-1]
            hi = hi[:, ::-1] if hi is not None else None
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return (t(lo), t(hi)) if hi is not None else (t(lo), torch.zeros(0))


# ---------------------------------------------------------------- train -----
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", required=True)
    ap.add_argument("--high", default=None)
    ap.add_argument("--zero_ref_only", action="store_true")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default=os.path.join(ROOT, "weights", "ZeroDCE++_LOL_finetuned.pth"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--scale_factor", type=int, default=1)
    ap.add_argument("--lambda_pair", type=float, default=1.0)
    ap.add_argument("--max_iters", type=int, default=0, help="debug: stop after N iterations")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    dev = torch.device("cuda" if (a.device != "cpu" and torch.cuda.is_available()) else "cpu")
    net = ZeroDCEpp().to(dev)
    sd = torch.load(a.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    print(f"loaded official weights {a.weights}; device={dev}")
    paired = (a.high is not None) and not a.zero_ref_only
    ds = LowLightSet(a.low, a.high if paired else None, a.size)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0, drop_last=len(ds) > a.bs)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lc, ls, le = L_color().to(dev), L_spa().to(dev), L_exp(16, 0.6).to(dev)
    log = []
    it = 0
    best = float("inf")
    for ep in range(a.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for lo, hi in dl:
            lo = lo.to(dev)
            enh, A = net(lo, a.scale_factor)
            loss = 1600 * L_tv(A) + ls(lo, enh) + 5 * lc(enh) + 10 * le(enh)
            if paired and hi.numel():
                hi = hi.to(dev)
                loss = loss + a.lambda_pair * (F.l1_loss(enh, hi) + ssim_loss(enh, hi))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            opt.step()
            tot += float(loss)
            it += 1
            if a.max_iters and it >= a.max_iters:
                break
        mean = tot / max(1, len(dl))
        log.append((ep, mean))
        print(f"epoch {ep + 1}/{a.epochs} loss {mean:.4f} ({time.time() - t0:.1f}s)")
        if mean < best:
            best = mean
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            torch.save(net.state_dict(), a.out)
        if a.max_iters and it >= a.max_iters:
            break
    with open(os.path.splitext(a.out)[0] + "_log.csv", "w") as f:
        f.write("epoch,loss\n" + "\n".join(f"{e + 1},{l:.6f}" for e, l in log))
    print(f"saved best checkpoint to {a.out}")


if __name__ == "__main__":
    main()

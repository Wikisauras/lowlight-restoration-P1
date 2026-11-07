"""
run_inference.py - Enhance every image in a folder.

  python run_inference.py --input_dir ./samples/ --output_dir ./results/ --mode auto
  python run_inference.py --input_dir ./samples/ --output_dir ./results/ --mode hybrid --save_intermediates
"""
import argparse
import json
import os
import time

import numpy as np

from src.evaluation import imread, imwrite, list_images, label_tile
from src.hybrid import enhance, MODES
from src.deep_learning import load_model, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default="results/inference")
    ap.add_argument("--mode", default="auto", choices=MODES)
    ap.add_argument("--fixed_params", action="store_true", help="disable adaptive parameters")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--save_intermediates", action="store_true")
    a = ap.parse_args()
    dev = str(get_device(a.device))
    model = load_model(a.weights, device=dev)
    files = list_images(a.input_dir)
    log = []
    for f in files:
        img = imread(f)
        t = time.perf_counter()
        y, info = enhance(img, a.mode, adaptive=not a.fixed_params, device=dev, model=model)
        ms = (time.perf_counter() - t) * 1000
        stem = os.path.splitext(os.path.basename(f))[0]
        imwrite(os.path.join(a.output_dir, f"{stem}.png"), y)
        if a.save_intermediates:
            for k, v in info.get("intermediates", {}).items():
                if isinstance(v, np.ndarray) and v.ndim in (2, 3) and v.shape[0] > 1:
                    v = v if v.ndim == 3 else np.repeat(v[..., None], 3, 2)
                    if k.endswith("curve_map"):
                        v = np.abs(v)
                    imwrite(os.path.join(a.output_dir, "intermediates", stem, f"{k}.png"), label_tile(v, k))
        s = info["stats"]
        log.append(dict(image=stem, condition=s.condition, flags=s.flags, chosen=info.get("chosen"), latency_ms=ms))
        print(f"{stem}: {s.condition} {s.flags} -> {info.get('chosen')} ({ms:.0f} ms)")
    with open(os.path.join(a.output_dir, "inference_log.json"), "w") as fh:
        json.dump(log, fh, indent=2)


if __name__ == "__main__":
    main()

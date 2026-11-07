"""
fill_report.py - Fill the development-run tables/figures into the report template.

  python scripts/fill_report.py --results results --template report/Report_P1_template.md --out report/Report_P1.md
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd

METHODS = ["raw", "clahe", "classical", "dl", "hybrid", "auto"]


def f(v, nd=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def method_table(s, ds, title, paired):
    d = s[s.dataset == ds].set_index("method").reindex([m for m in METHODS if m in set(s[s.dataset == ds].method)])
    if paired:
        h = "| Method | PSNR ↑ | SSIM ↑ | ΔE00 ↓ | NIQE ↓ | BRISQUE ↓ | Noise σ in→out | Noise amp ↓ | EPI ↑ | Clipped % ↓ | Halo | Latency ms (CPU) |"
        rows = [f"| {m} | {f(r.psnr)} | {f(r.ssim, 3)} | {f(r.delta_e)} | {f(r.niqe)} | {f(r.brisque)} | "
                f"{f(r.noise_in)}→{f(r.noise_out)} | {f(r.noise_amp)} | {f(r.epi, 3)} | {f(r.clipped_pct)} | {f(r.halo)} | {f(r.latency_ms, 0)} |"
                for m, r in d.iterrows()]
    else:
        h = "| Method | NIQE ↓ | BRISQUE ↓ | Noise σ in→out | Noise amp ↓ | Colour shift ΔE00 ↓ | Clipped % ↓ | Halo | Latency ms (CPU) |"
        rows = [f"| {m} | {f(r.niqe)} | {f(r.brisque)} | {f(r.noise_in)}→{f(r.noise_out)} | {f(r.noise_amp)} | "
                f"{f(r.colour_shift)} | {f(r.clipped_pct)} | {f(r.halo)} | {f(r.latency_ms, 0)} |" for m, r in d.iterrows()]
    sep = "|" + "---|" * (h.count("|") - 1)
    return f"**{title}**\n\n" + "\n".join([h, sep] + rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--template", default="report/Report_P1_template.md")
    ap.add_argument("--out", default="report/Report_P1.md")
    a = ap.parse_args()
    R = a.results
    figdir = os.path.join(os.path.dirname(a.out), "figures")
    os.makedirs(figdir, exist_ok=True)
    for p in glob.glob(os.path.join(R, "plots", "*.png")):
        shutil.copy(p, figdir)
    s = pd.read_csv(os.path.join(R, "summary.csv"))
    t = open(a.template, encoding="utf-8").read()

    t = t.replace("<<SYN_TABLE>>", method_table(s, "synthetic", "Synthetic paired set (50 pairs, 400×400)", True))
    t = t.replace("<<REAL_TABLE>>", method_table(s, "real_unseen", "Real unseen low-light photos (16 images, no GT)", False))

    cond = pd.read_csv(os.path.join(R, "summary_by_condition.csv"))
    piv = cond.pivot(index="condition", columns="method", values="psnr")
    piv_s = cond.pivot(index="condition", columns="method", values="ssim")
    cols = [m for m in METHODS if m in piv.columns]
    L = ["**Generalisation across illumination conditions (synthetic, PSNR / SSIM)**", "",
         "| Condition | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in piv.index:
        L.append(f"| {c.replace('_', ' ')} | " + " | ".join(f"{f(piv.loc[c, m])} / {f(piv_s.loc[c, m], 3)}" for m in cols) + " |")
    t = t.replace("<<COND_TABLE>>", "\n".join(L))

    lat = pd.read_csv(os.path.join(R, "latency.csv"))
    g = lat.groupby(["method", "device"]).latency_ms.median().unstack()
    res = lat.groupby("image")[["h", "w"]].first()
    L = [f"**Latency (median ms/image, development machine: 2-core cloud CPU, no GPU; images {int(res.h.median())}×{int(res.w.median())} median)** — GPU column: `[fill from results_real/latency.csv]`", "",
         "| Method | CPU (dev) | CPU (laptop) | GPU RTX 4060 |", "|---|---|---|---|"]
    for m in [x for x in METHODS[1:] if x in g.index]:
        L.append(f"| {m} | {f(g.loc[m, 'cpu'], 0)} | `[ ]` | `[ ]` |")
    t = t.replace("<<LAT_TABLE>>", "\n".join(L))

    syn = s[s.dataset == "synthetic"].set_index("method")
    real = s[s.dataset == "real_unseen"].set_index("method")
    best_psnr = syn.drop(index=["raw"], errors="ignore").psnr.idxmax()
    best_ssim = syn.drop(index=["raw"], errors="ignore").ssim.idxmax()
    notes = [
        f"> - Synthetic: best PSNR = **{best_psnr}** ({f(syn.loc[best_psnr, 'psnr'])} dB), best SSIM = **{best_ssim}** ({f(syn.loc[best_ssim, 'ssim'], 3)}); CLAHE baseline {f(syn.loc['clahe', 'psnr'])} dB / {f(syn.loc['clahe', 'ssim'], 3)}.",
        f"> - Noise amplification (synthetic): DL {f(syn.loc['dl', 'noise_amp'])}, classical {f(syn.loc['classical', 'noise_amp'])}, hybrid {f(syn.loc['hybrid', 'noise_amp'])} (CLAHE {f(syn.loc['clahe', 'noise_amp'])}).",
        f"> - NIQE on real unseen: input {f(real.loc['raw', 'niqe'])} → classical {f(real.loc['classical', 'niqe'])}, DL {f(real.loc['dl', 'niqe'])}, hybrid {f(real.loc['hybrid', 'niqe'])}.",
        f"> - Colour fidelity (ΔE00 vs GT, synthetic): classical {f(syn.loc['classical', 'delta_e'])}, DL {f(syn.loc['dl', 'delta_e'])}, hybrid {f(syn.loc['hybrid', 'delta_e'])}.",
        f"> - CPU latency: classical {f(syn.loc['classical', 'latency_ms'], 0)} ms, DL {f(syn.loc['dl', 'latency_ms'], 0)} ms, hybrid {f(syn.loc['hybrid', 'latency_ms'], 0)} ms per 400×400 image (DL time is mostly the quality-aware search + scoring, not the network).",
    ]
    t = t.replace("<<ANALYSIS_NOTES>>", "\n".join(notes))

    abl = pd.read_csv(os.path.join(R, "ablation.csv"))
    abl = abl[abl.dataset == "synthetic"].set_index("variant")
    from_order = [v for v in abl.index]
    L = ["| Variant | PSNR ↑ | SSIM ↑ | NIQE ↓ | Noise amp ↓ | Clipped % ↓ | Halo | ΔPSNR vs full |", "|---|---|---|---|---|---|---|---|"]
    for v in from_order:
        r = abl.loc[v]
        base = abl.loc[f"{v.split('/')[0]}/full", "psnr"] if f"{v.split('/')[0]}/full" in abl.index else abl.loc["hybrid/full", "psnr"]
        L.append(f"| {v} | {f(r.psnr)} | {f(r.ssim, 3)} | {f(r.niqe)} | {f(r.noise_amp)} | {f(r.clipped_pct)} | {f(r.halo)} | {r.psnr - base:+.2f} |")
    t = t.replace("<<ABL_TABLE>>", "\n".join(L))
    an = []
    for grp in ("classical", "hybrid"):
        full = abl.loc[f"{grp}/full"]
        rows = abl[[i.startswith(grp + "/") and i != f"{grp}/full" for i in abl.index]]
        d = (rows.psnr - full.psnr).sort_values()
        an.append(f"> - {grp}: largest PSNR drop when removing **{d.index[0].split('/')[1]}** ({d.iloc[0]:+.2f} dB); "
                  f"removing {d.index[-1].split('/')[1]} changes PSNR by {d.iloc[-1]:+.2f} dB.")
        fx = abl.loc[f"{grp}/fixed_params"]
        an.append(f"> - {grp}: fixed vs adaptive parameters: PSNR {f(fx.psnr)} vs {f(full.psnr)}, SSIM {f(fx.ssim, 3)} vs {f(full.ssim, 3)}, noise amp {f(fx.noise_amp)} vs {f(full.noise_amp)}.")
    if "dl/official_k1" in abl.index:
        an.append(f"> - DL-only {f(abl.loc['dl/official_k1', 'psnr'])} dB / {f(abl.loc['dl/official_k1', 'ssim'], 3)} → quality-aware DL {f(abl.loc['dl/quality_aware', 'psnr'])} dB / {f(abl.loc['dl/quality_aware', 'ssim'], 3)} → hybrid {f(abl.loc['hybrid/full', 'psnr'])} dB / {f(abl.loc['hybrid/full', 'ssim'], 3)}.")
    if "hybrid/-prior" in abl.index:
        an.append(f"> - Illumination prior on/off: PSNR {f(abl.loc['hybrid/full', 'psnr'])} vs {f(abl.loc['hybrid/-prior', 'psnr'])}; clipped % {f(abl.loc['hybrid/full', 'clipped_pct'], 3)} vs {f(abl.loc['hybrid/-prior', 'clipped_pct'], 3)}.")
    t = t.replace("<<ABL_NOTES>>", "\n".join(an))

    fl = pd.read_csv(os.path.join(R, "failures.csv"))
    fh = fl[fl.method == "hybrid"].groupby("metric").head(2)
    L = ["| Metric (hybrid) | Worst image | Condition | Value |", "|---|---|---|---|"]
    for _, r in fh.iterrows():
        L.append(f"| {r.metric} | {r.dataset}/{r.image} | {r.condition} | {f(r.value)} |")
    t = t.replace("<<FAIL_TABLE>>", "\n".join(L))
    fimgs = sorted(glob.glob(os.path.join(R, "failures", "hybrid_*.png")))
    pick = [p for p in fimgs if "noise_amp" in p or "colour_shift" in p][:1] + [p for p in fimgs if "halo" in p or "clipped" in p][:1]
    for i, p in enumerate(pick, 1):
        shutil.copy(p, os.path.join(figdir, f"failure_{i}.png"))
        t = t.replace(f"![failure {i}](figures/failure_{i}.png)",
                      f"![failure {i}](figures/failure_{i}.png)\n\n<span class='caption'>Worst case: {os.path.basename(p)[7:-4]} (input | classical | DL | hybrid)</span>")
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write(t)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

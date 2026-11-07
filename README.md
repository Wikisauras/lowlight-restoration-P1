# Robust Low-Light Image Restoration & Enhancement (Problem P1)

Illumination-adaptive low-light restoration that combines a classical Retinex pipeline
with the Zero-DCE++ network (official pretrained weights) in a hybrid fusion framework,
plus a full evaluation / ablation / failure-analysis suite and a Streamlit app.

**Live demo:** https://atreides-lowlight.streamlit.app (Streamlit Community Cloud, CPU. The first load after idle takes ~1 min to wake up.)
**Team Atreides (RV University, CIE-3):** Vansh Anand Raval (classical), Varun G Gowda (hybrid + integration),
V Sreeshanth Reddy (deep learning), Vedagya Sharma (analysis + app), Yuvrat Mittal (evaluation).

### Results on LOL eval15 (15 real paired test images, CPU)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise amp ↓ | Latency (ms) |
|---|---|---|---|---|---|---|
| CLAHE baseline | 9.37 | 0.364 | 0.501 | 7.52 | 1.40 | 10 |
| Classical | 17.21 | 0.739 | 0.355 | 5.96 | 0.25 | 286 |
| Zero-DCE++ (official weights) | 17.22 | 0.549 | 0.376 | 7.93 | 0.89 | 1539 |
| **Hybrid** | 17.62 | 0.741 | 0.360 | 6.23 | 0.21 | 1945 |
| Hybrid, auto mode | **17.68** | **0.742** | 0.359 | 6.12 | 0.21 | 2820 |

**LOL fine-tuning** (`src/train.py`, 485 our485 pairs, CPU, best checkpoint at epoch 4, loss plateaued):
Zero-DCE++ 17.22 → 17.26 dB, ΔE00 15.23 → 14.64 (better colour), SSIM 0.549 → 0.541; hybrid 17.62 → 17.49 dB.
Fine-tuning gives no clear gain on PSNR/SSIM, so the official weights stay the default; the checkpoint is
`weights/ZeroDCE++_LOL_finetuned.pth` (use `--weights` in `run_inference.py` / `src.evaluation`).

Full tables, ablation and failure analysis: `report/Report_P1.pdf`.

```
requirements.txt            CPU torch + all packages (also used by Streamlit Cloud)
packages.txt                system libraries for Streamlit Cloud (OpenCV)
run_inference.py            folder -> enhanced folder (CLI)
src/analysis.py             image-condition analysis -> ALL adaptive parameters
src/classical.py            LIME illumination + guided filter, Retinex/MSRCR, CLAHE, denoise, colour, highlights
src/deep_learning.py        Zero-DCE++ (official weights) + quality-aware curve-strength selection
src/train.py                fine-tuning of Zero-DCE++ on LOL (paired + zero-reference losses)
src/hybrid.py               illumination prior -> DL, per-pixel fusion, guards, auto mode, enhance()
src/evaluation.py           metrics, 5-way comparison, ablation, failure analysis, CSV + plots
app/app.py                  Streamlit application
scripts/get_lol_eval15.py   download only the 15 LOL test pairs (HTTP range requests, a few MB)
scripts/get_lol_train.py    download the 485 LOL training pairs (for fine-tuning)
scripts/make_synthetic.py   paired synthetic test set (5 illumination conditions)
scripts/run_full_eval.bat   ONE command for the full evaluation on the real datasets
scripts/finetune_lol.bat    optional LOL fine-tuning + re-evaluation
scripts/md_to_pdf.py        render the report markdown to PDF
tests/test_edge_cases.py    black / saturated / tiny / huge / grayscale / 16-bit inputs
weights/ZeroDCE++_Epoch99.pth   official Zero-DCE++ checkpoint (Li et al., TPAMI 2021), default
weights/ZeroDCE++_LOL_finetuned.pth   our LOL fine-tuned checkpoint
samples/                    a few test images for a quick try
report/                     Report_P1.md / .pdf (final numbers), figures, fonts
scripts/fill_report.py      regenerates the development tables in the report from results/
scripts/make_figures.py     intermediate-representation figures
```

## 1. Setup (Windows, Python 3.11)

```bat
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt                                 &REM installs CPU torch
python -m pytest tests -q                                       &REM edge-case tests
```
For an NVIDIA GPU, install the CUDA build first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
The first run of NIQE / BRISQUE / LPIPS downloads small metric weights (pyiqa,
AlexNet), so you need internet access for that run.

## 2. Datasets

| Dataset | Type | Use | Where to put it |
|---|---|---|---|
| LOL v1 (485 train / 15 test pairs) | paired | PSNR, SSIM, LPIPS, ΔE; fine-tuning | `data/LOL/our485/{low,high}`, `data/LOL/eval15/{low,high}` |
| DICM, LIME, MEF, NPE, VV | unpaired | unseen-image test (NIQE, BRISQUE, failures) | `data/unpaired/DICM/`, `data/unpaired/LIME/`, ... |
| synthetic (included) | paired | per-condition generalisation | `data/synthetic/{low,high}` |
| real_unseen (included) | unpaired | 16 real low-light photos (Zero-DCE++ repo test set, one per scene) | `data/real_unseen/` |

Sources: LOL - https://huggingface.co/datasets/geekyrakshit/LoL-Dataset. The quickest way is
`python scripts\get_lol_eval15.py` (test split) and `python scripts\get_lol_train.py` (training split).
Datasets are not stored in this repository. DICM/LIME/MEF/NPE/VV - Google Drive link in the README of
https://github.com/baidut/BIMEF. Missing folders are skipped automatically.

## 3. Run

```bat
:: evaluation used for the report (LOL eval15 + real unseen photos, CPU)
python -m src.evaluation --paired LOL=data/LOL/eval15 --unpaired_single real_unseen=data/real_unseen --out results_real --ablation --ablation_limit 10 --latency_n 5 --device cpu

:: full evaluation including the unpaired benchmark sets (if present)
scripts\run_full_eval.bat               -> results_real\

:: app
streamlit run app/app.py

:: batch inference
python run_inference.py --input_dir ./samples/ --output_dir ./results/ --mode auto
python run_inference.py --input_dir ./samples/ --output_dir ./results/ --mode hybrid --save_intermediates

:: optional: fine-tune Zero-DCE++ on LOL, then evaluate with the new weights
scripts\finetune_lol.bat
```

Outputs of the evaluation (`results_real/`):
`per_image.csv`, `summary.csv`, `summary_by_condition.csv`, `ablation.csv`,
`latency.csv` (CPU and GPU), `failures.csv`, `failures/*.png` (worst cases),
`plots/*.png`, and **`report_tables.md`**, with the markdown tables ready to paste
into the report.

### Deployment

The app is deployed on Streamlit Community Cloud straight from this repository
(main file `app/app.py`; `requirements.txt` installs CPU torch, `packages.txt` the OpenCV system libraries).
To redeploy: share.streamlit.io → Create app → this repo → `app/app.py` → Deploy.

## 4. Method summary

1. **Analysis** (`analysis.py`): brightness, entropy, noise σ (Immerkær), illumination
   non-uniformity (CoV of the smoothed max-RGB map), colour cast (a*b* distance), and
   dark/bright/clipped fractions. Every parameter comes from these, for example the
   LIME gamma `g = 1 - ln(0.45)/ln(mean T)`, guided-filter eps `(2·σ·gain)²`, the CLAHE
   clip from entropy and noise, MSRCR weight from non-uniformity and noise, and the fusion
   exponents. `FIXED_PARAMS` exists only for the ablation.
2. **Classical** (`classical.py`): gray-world → max-RGB illumination → guided-filter
   refinement → Retinex relighting `I/T^g` (+ adaptive MSRCR blend) → guided-filter
   denoising in YCrCb, stronger in dark regions → CLAHE on LAB L → highlight protection.
3. **Deep learning** (`deep_learning.py`): Zero-DCE++ with the official `Epoch99.pth`
   (verified against the official code: mean abs. difference 0.0009). Quality-aware
   strategy: the curve map A is predicted once, and strengths k·A are scored with a
   no-reference score (NIQE/BRISQUE optional) to pick the best.
4. **Hybrid** (`hybrid.py`): the classical illumination map attenuates the Zero-DCE++
   curves in already-lit regions (illumination prior). The DL output is then denoised
   with the classical guided filter. Classical and DL candidates are fused per pixel using
   well-exposedness × local contrast × saturation × (1/local noise) weights in a
   Laplacian pyramid, followed by dark-region denoising and a no-blow-out guard.
   **Auto** mode runs classical, DL and hybrid and selects the best by a no-reference score.

## 5. Notes and limitations

* LPIPS downloads the torchvision AlexNet weights on first use (internet needed once).
* All final numbers were measured on CPU. GPU latency was not measured.
* Very dark JPEG inputs show amplified 8×8 block artefacts and colour noise. This is
  documented in the failure analysis.
* `data/real_unseen` comes from the Zero-DCE++ repository's test folder. The Zero-DCE++
  weights and architecture are © the original authors (Li, Guo, Loy). All other code is ours.

# Robust Low-Light Image Restoration & Enhancement (Problem P1)

Illumination-adaptive low-light restoration that combines a classical Retinex pipeline
with the Zero-DCE++ network (official pretrained weights) in a hybrid fusion framework,
plus a full evaluation / ablation / failure-analysis suite and a Streamlit app.

```
requirements.txt
run_inference.py            folder -> enhanced folder (CLI)
src/analysis.py             image-condition analysis -> ALL adaptive parameters
src/classical.py            LIME illumination + guided filter, Retinex/MSRCR, CLAHE, denoise, colour, highlights
src/deep_learning.py        Zero-DCE++ (official weights) + quality-aware curve-strength selection
src/train.py                optional fine-tuning of Zero-DCE++ on LOL
src/hybrid.py               illumination prior -> DL, per-pixel fusion, guards, auto mode, enhance()
src/evaluation.py           metrics, 5-way comparison, ablation, failure analysis, CSV + plots
app/app.py                  Streamlit application
scripts/make_synthetic.py   paired synthetic test set (5 illumination conditions)
scripts/run_full_eval.bat   ONE command for the full evaluation on the real datasets
scripts/finetune_lol.bat    optional LOL fine-tuning + re-evaluation
scripts/md_to_pdf.py        render the report markdown to PDF
tests/test_edge_cases.py    black / saturated / tiny / huge / grayscale / 16-bit inputs
weights/ZeroDCE++_Epoch99.pth   official Zero-DCE++ checkpoint (Li et al., TPAMI 2021)
data/synthetic, data/real_unseen   test data used in development (see below)
results/                    evaluation outputs from the development run (CPU)
report/                     Report_P1.md / .pdf (development numbers), Report_P1_template.md, figures, fonts
scripts/fill_report.py      regenerates the development tables in the report from results/
scripts/make_figures.py     intermediate-representation figures
```

## 1. Setup (Windows 11, Python 3.11, RTX 4060)

```bat
py -3.11 -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     &REM should print True
python -m pytest tests -q                                       &REM edge-case tests
```
The first run of NIQE / BRISQUE / LPIPS downloads small metric weights (pyiqa,
AlexNet), so you need internet access for that run.

## 2. Datasets

| Dataset | Type | Use | Where to put it |
|---|---|---|---|
| LOL v1 (485 train / 15 test pairs) | paired | PSNR, SSIM, LPIPS, ΔE; fine-tuning | `data/LOL/our485/{low,high}`, `data/LOL/eval15/{low,high}` |
| DICM, LIME, MEF, NPE, VV | unpaired | unseen-image test (NIQE, BRISQUE, failures) | `data/unpaired/DICM/`, `data/unpaired/LIME/`, ... |
| synthetic (included) | paired | per-condition generalisation | `data/synthetic/{low,high}` |
| real_unseen (included) | unpaired | 16 real low-light photos (Zero-DCE++ repo test set, one per scene) | `data/real_unseen/` |

Sources: LOL - https://huggingface.co/datasets/geekyrakshit/LoL-Dataset (or a Kaggle
"LOL dataset" mirror). DICM/LIME/MEF/NPE/VV - Google Drive link in the README of
https://github.com/baidut/BIMEF. Missing folders are skipped automatically.

## 3. Run

```bat
:: full evaluation (all metrics, comparison, ablation, failure analysis, plots)
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

### Filling the report with your LOL / DICM / LIME / MEF / NPE numbers

1. Run `scripts\run_full_eval.bat` (and optionally `scripts\finetune_lol.bat`).
2. Open `results_real\report_tables.md` and copy the LOL, unpaired-dataset, ablation and latency tables
   (CPU and GPU columns) into `report\Report_P1.md`, replacing the `[ ]` placeholders in section 3.1 and 3.3.
   Do **not** re-run `fill_report.py` on `results_real` (it is written for the development folder `results/`).
3. Copy `results_real\plots\*.png` and 2-3 images from `results_real\failures\` into `report\figures\`, then
   `python scripts/md_to_pdf.py report/Report_P1.md report/Report_P1.pdf`.

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

* LPIPS needs the torchvision AlexNet weights, which could not be downloaded in the
  development sandbox. It is computed automatically on any machine with internet.
* Very dark JPEG inputs show amplified 8×8 block artefacts and colour noise. This is
  documented in the failure analysis.
* `data/real_unseen` comes from the Zero-DCE++ repository's test folder. The Zero-DCE++
  weights and architecture are © the original authors (Li, Guo, Loy). All other code is ours.

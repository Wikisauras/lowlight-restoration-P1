# Hackathon Technical Solution Report

| Field | Details |
| :--- | :--- |
| **Problem ID** | P1 |
| **Problem Title** | Robust Low-Light Image Restoration and Enhancement |
| **Team Name** | Atreides |
| **Team Members** | Vansh Anand Raval (1RUA24CSE0513), Varun G Gowda (1RUA24CSE0522), V Sreeshanth Reddy (1RUA24CSE0523), Vedagya Sharma (1RUA24CSE0524), Yuvrat Mittal (1RUA24CSE0552) |
| **GitHub Repository** | `[Public GitHub URL]` — commit `[hash]` |
| **Live Demo / Deployment** | Streamlit app, run locally with `streamlit run app/app.py` `[+ Streamlit Cloud link if deployed]` |

**Team contributions**

| Member | USN | Module / responsibility |
| :--- | :--- | :--- |
| Vansh Anand Raval | 1RUA24CSE0513 | Classical pipeline (`src/classical.py`): illumination estimation, Retinex/MSRCR, CLAHE, guided-filter denoising, highlight protection |
| Varun G Gowda | 1RUA24CSE0522 | Hybrid integration (`src/hybrid.py`): illumination prior, fusion, guards, auto mode; report lead |
| V Sreeshanth Reddy | 1RUA24CSE0523 | Deep learning (`src/deep_learning.py`, `src/train.py`): Zero-DCE++, quality-aware strategy, LOL fine-tuning |
| Vedagya Sharma | 1RUA24CSE0524 | Image analysis & adaptive parameters (`src/analysis.py`); Streamlit app (`app/app.py`) |
| Yuvrat Mittal | 1RUA24CSE0552 | Evaluation (`src/evaluation.py`): metrics, ablation, failure analysis; synthetic test set |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:**
    * Severe under-exposure: mean luminance from below 0.02 (extreme dark) to around 0.2, so brightness gains of ×3 to ×30 are needed.
    * Sensor noise grows with the gain: shot and read noise, plus 8-bit quantisation and JPEG blocks in dark regions.
    * Colour distortion: casts from the light source, and channel imbalance near black.
    * Non-uniform lighting: lamps, windows and backlight next to deep shadow, where one global curve either blows out the light or leaves the shadow dark.
    * Failure modes we target: over-exposure/clipping, halos at light/dark borders, colour shift, noise amplification, texture/edge loss.
    * Constraints: runs on a laptop (RTX 4060, 8 GB) and on CPU, with no paired data needed at inference time.

> ✍️ **WRITE IN YOUR OWN WORDS — Proposed Methodology (2–3 sentences, owner: Varun).** Points to cover:
> (1) the image is analysed first, and the measured brightness, noise, non-uniformity and colour cast set every parameter;
> (2) two complementary enhancers: a classical Retinex pipeline (good noise control, physically meaningful illumination) and Zero-DCE++ (learned, local, natural curves);
> (3) the hybrid uses the classical illumination map to steer the network and fuses both per pixel, so a static baseline cannot over-enhance lit regions or amplify noise in shadows.

---

## 2. Technical Architecture & Implementation

```
input -> guards (gray->3ch, size cap, black check) -> analysis.py (stats -> parameters)
      -> classical.py --------------------------> classical output --+
      -> illumination map T --> prior m ----+                        |--> per-pixel weights -> Laplacian
      -> deep_learning.py (Zero-DCE++ A) ---+--> k*m*A curves -> DL output -> classical denoise --+     pyramid fusion
                                                                                                  -> dark-region denoise -> no-blow-out guard -> output
auto mode: runs classical / DL / hybrid and picks the best by a no-reference score
```

### 2.1 Classical Algorithmic Baseline
* **Algorithms Utilized** (`src/classical.py`, in order):

| # | Stage | Method |
| :-- | :--- | :--- |
| 1 | Colour correction | Gray-world gains on reliable pixels (0.02 < Y, max < 0.98), blended by an adaptive strength |
| 2 | Illumination estimation | Max-RGB initial map T₀ (LIME) |
| 3 | Illumination refinement | Edge-aware guided filter (own implementation), T ≥ 0.85·T₀ |
| 4 | Retinex decomposition | Reflectance R = I / T; relit = I / T^γ |
| 5 | Retinex detail | MSRCR (3 scales, colour restoration, 1 %/99 % balance), brightness-matched and blended with weight w_msr |
| 6 | Contrast | CLAHE on CIELAB L* (16-bit) |
| 7 | Edge-preserving denoising | Guided filter in YCrCb (2 passes for strong noise; chroma with 2× window), blended by a dark-region mask (1−T)^(1/p). Runs after CLAHE, using the noise measured after CLAHE |
| 8 | Highlight protection | Already-lit pixels kept near their input value; soft-knee roll-off on the rest |

* **Adaptive Selection Logic** (`src/analysis.py`; *t* = mean of the max-RGB map, σ = Immerkær noise estimate, σ_q = 8-bit quantisation noise):

| Parameter | Rule | Reason |
| :--- | :--- | :--- |
| Relighting γ | γ = 1 − ln(0.45) / ln(t), clipped to [0.15, 0.92] | maps the mean illumination to mid-grey exposure |
| Expected noise | σ_post = √(σ² + σ_q²) · t^(−γ) | noise after the gain decides the denoising strength |
| Guided-filter radius / eps (illumination) | r ∝ image size, eps 1e-3 (4e-3 if non-uniform) | smoother T for strong lighting gradients |
| MSRCR scales / weight | σ_k = {15, 80, 250}·(size/512); w = 0.15 + 0.5·CoV − 3·σ_post | MSRCR helps uneven light but amplifies noise |
| Denoise strength / eps | s = σ_post / 0.04; eps = (2σ_post)², chroma (4σ_post)²; radius 2 + 6s | stronger filtering only when there is more noise |
| CLAHE clip / tiles | clip = 1.2 + 2.6(1 − H/8) − 20σ_post; tiles = min side / 96 (×1.5 if non-uniform) | more contrast for low-entropy images, less when noisy |
| Colour-correction strength | (cast − 2) / 10, clipped to [0, 0.9] | only correct a measurable cast |
| Highlight knee | 0.85 − 0.6·(bright fraction) | earlier roll-off when much of the frame is bright |
| Fusion exponents | exposure 1 (1.5 if non-uniform), contrast 1 (0.6 if H > 7), saturation 1 (0.5 if cast), noise 0.5 + 2s | see 2.3 |
| DL strengths tried / curve scale | {0.6…1.4} if very dark, else {0.4…1.0}; scale = size/256 | see 2.2 |

All parameters come from these rules. The `FIXED_PARAMS` set (γ 0.8, CLAHE 2.0/8×8, etc.) is used only in the "fixed vs adaptive" ablation.

### 2.2 Deep Learning Pipeline
* **Model Architecture:** Zero-DCE++ (Li, Guo & Loy, TPAMI 2021) with the official pretrained checkpoint `Epoch99.pth` (≈10 k parameters). It has 7 depthwise-separable conv layers, predicts a per-pixel 3-channel curve map A ∈ [−1, 1], and applies 8 iterations of LE(x) = x + A·(x² − x). Our re-implementation loads the official weights with `strict=True` and matches the official inference code (mean absolute difference 0.0009 on a test image).

> ✍️ **WRITE IN YOUR OWN WORDS — Selection rationale (owner: Sreeshanth).** Notes: zero-reference (no paired data needed), so it generalises to unseen scenes; very small and fast; the curve formulation is monotonic, so it cannot invent content; and the curve map A can be modulated by our classical illumination prior (this is what makes the hybrid possible). Compare briefly with RetinexNet (needs pairs, noisy outputs) and EnlightenGAN (heavier, GAN artefacts).

* **Image-quality-aware strategy:** A is predicted once, and the curves are re-applied with strengths k·A for several k on a ≤ 384 px copy. Each candidate gets a no-reference score: exposure closeness to 0.5 + entropy/8 − 4·(new clipping) − 6·(noise σ) − 0.5·(colour cast). NIQE/BRISQUE can be used as the scorer instead (`scorer="niqe"`). The best k is then applied at full resolution.
* **Loss Functions & Training** (`src/train.py`, optional fine-tuning from the official weights):

| Term | Weight | Purpose |
| :--- | :--- | :--- |
| Illumination smoothness L_TV(A) | 1600 | smooth curve maps |
| Spatial consistency L_spa | 1 | keeps neighbouring contrast (edges) |
| Colour constancy L_col | 5 | gray-world colour balance |
| Exposure control L_exp (E = 0.6, 16×16 patches) | 10 | well-exposed output |
| Paired L1 + (1 − SSIM) vs LOL ground truth (our addition) | λ = 1 | uses LOL pairs when available |

    Data: LOL our485 (485 pairs, 256×256 random crops, flips), Adam lr 1e-4, batch 8, 30 epochs. Command: `scripts\finetune_lol.bat`.
    Result: `[fine-tuned vs official LOL PSNR/SSIM — fill from results_finetuned/summary.csv, or state "not run"]`.

### 2.3 Hybrid Integration & Adaptive Mechanism
* **Fusion Strategy** (`src/hybrid.py`):
    1. **Illumination prior → network:** the classical T is smoothed to a low-frequency lighting map T_s, and the Zero-DCE++ curve map becomes A′ = m·k·A with m = 1 − 0.85·clip((T_s − τ)/0.35, 0, 1)^(1/β), where τ = max(mean + std of T_s, 0.35). Lit regions (lamps, windows, sky) get weaker curves. Uniformly dark scenes keep m = 1.
    2. **Classical denoising of the DL output:** guided-filter eps is set from the DL candidate's own measured gain.
    3. **Per-pixel weights** for each candidate (classical, DL): W = E^a · C^b · S^c · N^d, where E = well-exposedness exp(−(x−0.5)²/2·0.2²), C = |Laplacian| of the guided-smoothed luminance (structure, not noise), S = channel std (saturation), and N = 1/(1 + local noise/σ_post). Exponents a–d come from 2.1.
    4. **Laplacian-pyramid fusion** (Burt–Adelson / Mertens) avoids seams and halos.
    5. **Dark-region denoising** with strength ∝ (1 − T) and the measured residual noise.
    6. **No-blow-out guard:** where T_s is high, output luminance ≤ 1.08·input + 0.03; elsewhere a soft knee.

> ✍️ **WRITE IN YOUR OWN WORDS — Decision Logic (owner: Varun).** Explain how the four weights trade amplification against noise: dark noisy regions → the noise weight prefers the better-denoised candidate, and dark-region denoising is strongest there; lit regions → the prior weakens the DL curves and the guard caps luminance, so they are not blown out; flat regions vs edges → the contrast weight is computed on smoothed luminance so noise is not rewarded. Also explain **auto mode**: classical, DL and hybrid all run, the same no-reference score picks one (+0.05 bias to hybrid), and a completely black frame is returned unchanged ("no recoverable signal").

---

## 3. Experimental Benchmarks & Evaluation

**Protocol.** Methods: raw input, CLAHE baseline (fixed 2.0 / 8×8), classical, DL (official Zero-DCE++, k = 1), hybrid, auto. Paired metrics: PSNR, SSIM, LPIPS (AlexNet), CIEDE2000 vs ground truth. No-reference metrics: NIQE, BRISQUE (pyiqa). Efficiency: median latency, CPU and GPU. Noise: Immerkær σ before/after, and amplification = σ_out / (σ_in · brightness gain), where < 1 means noise is suppressed relative to the signal gain. Edge preservation: EPI (correlation of Laplacians) vs ground truth.
Datasets: **LOL eval15** (paired, real); **DICM, LIME, MEF, NPE, VV** (unpaired, *previously unseen*); **synthetic** (50 pairs, 5 conditions: physically based darkening in linear light + Poisson-Gaussian noise + colour cast / spotlight / light sources); **real_unseen** (16 real low-light photos).
Command for everything: `scripts\run_full_eval.bat` (tables are written to `results_real/report_tables.md`).

### 3.1 Quantitative Benchmarks

**LOL eval15 (real paired):**

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual Metric (LPIPS / NIQE) ↓ | Latency (ms / frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Classical Baseline** | `[0.00]` | `[0.00]` | `[0.00 / 0.00]` | `[00 ms]` |
| **Deep Learning Only** | `[0.00]` | `[0.00]` | `[0.00 / 0.00]` | `[00 ms]` |
| **Proposed Hybrid System** | `[0.00]` | `[0.00]` | `[0.00 / 0.00]` | `[00 ms]` |

*(Copy from `results_real/report_tables.md`, section LOL. Latency: GPU median from `latency.csv`.)*

**Unseen unpaired datasets (NIQE ↓ / BRISQUE ↓):**

| Method | DICM | LIME | MEF | NPE | VV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CLAHE | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Classical | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| DL (Zero-DCE++) | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Hybrid | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Development results (measured on CPU during development; LPIPS not available in that environment):**

<<SYN_TABLE>>

<<REAL_TABLE>>

<<COND_TABLE>>

<<LAT_TABLE>>

![comparison](figures/comparison_synthetic.png)

![per-condition](figures/per_condition_psnr.png)

![noise](figures/noise.png)

> ✍️ **WRITE IN YOUR OWN WORDS — Analysis (2–3 sentences, owner: Yuvrat).** Facts from the development run to interpret (update with LOL):
>
<<ANALYSIS_NOTES>>

### 3.2 Visual Outputs & Intermediate Representations

1. **Extreme low-light** (raw input vs enhanced output; rows: input | CLAHE | classical | DL | hybrid | ground truth):

![examples synthetic](figures/examples_synthetic.png)

![examples real](figures/examples_real_unseen.png)

2. **Non-uniform / backlit scenes**: spotlight scene. The illumination prior (tile 9) lowers the network's curve strength inside the lit area. Note that on our synthetic set this changes PSNR by only about 0.02 dB (see 3.3), so we do not claim a measured gain from it:

![non-uniform intermediates](figures/intermediates_non_uniform__020.png)

3. **Intermediate states** (real image): illumination map, reflectance, Retinex, denoised, CLAHE, curve map, prior, fusion weights, final:

![real intermediates](figures/intermediates_1021_0_.png)

### 3.3 Ablation Study
Each component is removed in turn (the rest unchanged), fixed vs adaptive parameters, DL-only vs hybrid, with vs without the illumination prior. Development numbers (15 synthetic pairs spread over all 5 conditions):

<<ABL_TABLE>>

![ablation](figures/ablation.png)

LOL ablation: `[paste from results_real/report_tables.md → Ablation]`

> ✍️ **WRITE IN YOUR OWN WORDS — Ablation discussion (owner: each module owner writes one line for their components).** Facts:
>
<<ABL_NOTES>>
> - Removing denoising is the most harmful change for structure and noise: classical SSIM 0.662 → 0.511 and noise amplification 0.31 → 1.45 (halo score 0.67 → 1.31).
> - Mixed results (important to state honestly): on this small ablation (15 images, 3 per condition) removing the illumination stage, MSRCR, or the exposure weight *raises* PSNR, and the quality-aware strength search lowers DL PSNR by 0.12 dB. Removing the illumination stage lowers SSIM (0.662 → 0.637) and worsens NIQE (6.53 → 7.01). Adaptive parameters win on PSNR but the fixed-parameter hybrid has a slightly higher SSIM and a better NIQE (6.47 vs 7.08). The differences are small and the sample is small, so these components should be re-checked on LOL before being claimed as beneficial.
> - Illumination prior by condition (PSNR with / without): non-uniform 15.73 / 15.71, bright spots 14.17 / 14.17, all others identical. Zero-DCE++ did not blow out the lit regions in our synthetic spotlights, so the attenuating prior had nothing to correct.
> - Auto mode picked the hybrid output for all 50 synthetic images and 15 of the 16 real photos (DL for `435_0_`), so its numbers are almost identical to the hybrid's. It adds latency without much benefit on this data.

---

## 4. Failure Modes & Limitations

Automated failure analysis (`results*/failures.csv`, images in `results*/failures/`): for classical, DL and hybrid, the worst three images are listed for each of % newly clipped pixels, colour shift (ΔE00), halo score and noise amplification.

<<FAIL_TABLE>>

![failure 1](figures/failure_1.png)

![failure 2](figures/failure_2.png)

* **Observed Failure Cases** (facts from the development run):
    * **Over-smoothing / loss of fine detail in extremely dark inputs.** In the extreme-dark synthetic rows of the example grid, classical and hybrid outputs are clean but visibly blurred and slightly hazy (low saturation). Plain DL keeps more texture but is very noisy. On the synthetic set noise amplification is 0.33 (hybrid) and 0.35 (classical) vs 1.04 (DL), but edge preservation (EPI vs ground truth) is lower for the denoised outputs: 0.44 (hybrid), 0.50 (classical), 0.42 (DL). So noise suppression is bought partly with detail.
    * **Colour shift on very dark real photos.** Hybrid colour shift vs the input chroma (ΔE00) averages 10.7 on the 16 real photos and peaks at 20.8 (`450_0_`) and 18.0 (`126_0_`). In `126_0_` (mean luminance about 0.01) a strong green cast appears.
    * **Compressed-image artefacts.** In the darkest JPEG/PNG-from-JPEG inputs, 8×8 block structure and chroma noise are amplified by a gain of about 25–30 (visible in the failure image above).
    * **Noise amplification above 1** only occurred once on real data: `435_0_` (1.54), an image that is not very dark (mean luminance 0.23), where the gain is small and the "noise" estimate is dominated by texture.
    * **CLAHE ordering.** CLAHE re-amplifies residual noise. Moving denoising after CLAHE changed classical noise amplification from 0.68 to 0.35 and SSIM from 0.675 to 0.689 (development subset).
    * **Illumination prior.** A symmetric version (boosting shadows as well as attenuating lights) lowered PSNR on every condition, because Zero-DCE++ curves already saturate in dark regions. Only the attenuating version is kept, and it makes almost no measurable difference on our synthetic data.
    * No over-exposure failure was found: newly clipped pixels were at most 0.02 % for hybrid on any real image, and 0.00 % on synthetic images with saturated light sources.

> ✍️ **WRITE IN YOUR OWN WORDS — Root Cause & Mitigation (owner: Vansh + Sreeshanth).** Notes: blocks — the gain multiplies the quantisation step (1/255 → ~0.1), and the guided filter keeps block edges because they look like real edges; mitigation: deblocking before relighting, or a noise model including σ_q (we added the σ_q floor). Colour noise — near zero, channels clip at 0, so noise is no longer zero-mean (positive bias → colour cast); mitigation: estimate in linear/RAW space, or chroma NLM. Warm light — gray-world assumes a neutral average; mitigation: limit the strength for warm casts, or use a learned white balance. Prior — explain why a multiplicative boost beyond A = 1 is clipped and only shifts mid-tones.

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands
```bash
# 1. Clone the repository and install dependencies
git clone <your-github-repo-url>
cd lowlight_project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Run inference on a sample directory
python run_inference.py --input_dir ./samples/ --output_dir ./results/ --mode auto

# 3. Streamlit app
streamlit run app/app.py

# 4. Full evaluation (LOL + DICM/LIME/MEF/NPE/VV + synthetic, ablation, failure analysis)
scripts\run_full_eval.bat
```

### Deployment Format
* Streamlit dashboard (`app/app.py`): upload, mode (classical / DL / hybrid / auto), adaptive on/off, CPU/GPU, detected-condition report, all intermediate maps, before/after slider, metrics (latency, noise σ before→after, clipping, ΔE, halo, EPI, NIQE/BRISQUE, plus PSNR/SSIM/LPIPS if a ground truth is uploaded), download button.
* CLI: `run_inference.py`. Library: `from src.hybrid import enhance`.
* Edge cases handled and unit-tested (`tests/test_edge_cases.py`, 8 tests): completely black frames (returned unchanged and flagged), saturated light sources (preserved), 1×1 to 4000×3000 inputs, grayscale / 16-bit / float / RGBA inputs.

---

### Student Submission Checklist
- [ ] Working GitHub repository link included with an updated `README.md`.
- [x] No raw code pasted into this report.
- [ ] All table metrics match test runs reproducible via the repository.
- [x] All team member details accurately recorded.

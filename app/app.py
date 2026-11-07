"""
Streamlit app - Robust Low-Light Image Restoration (Problem P1).

Run locally:   streamlit run app/app.py
"""
import base64
import io
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import analysis as A                      # noqa: E402
from src.hybrid import enhance                     # noqa: E402
from src.deep_learning import load_model, get_device  # noqa: E402
from src import evaluation as E                    # noqa: E402

st.set_page_config(page_title="Low-Light Restoration (P1)", layout="wide")


# ----------------------------------------------------------------------------- helpers
@st.cache_resource
def get_model(device):
    return load_model(device=device)


@st.cache_resource
def get_iqa():
    return E.IQA(device="cpu")


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def colormap(m, cmap=cv2.COLORMAP_INFERNO):
    m = np.asarray(m, np.float32)
    m = (m - m.min()) / max(m.max() - m.min(), 1e-6)
    return cv2.cvtColor(cv2.applyColorMap((m * 255).astype(np.uint8), cmap), cv2.COLOR_BGR2RGB)


def b64(img_u8):
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def before_after_slider(before, after, height=520):
    """Self-contained HTML before/after slider (no extra dependency)."""
    h, w = before.shape[:2]
    disp_h = min(height, int(900 * h / w))
    html = f"""
    <div id="c" style="position:relative;width:100%;max-width:{int(disp_h * w / h)}px;aspect-ratio:{w}/{h};
         overflow:hidden;border-radius:8px;user-select:none;cursor:ew-resize">
      <img src="data:image/jpeg;base64,{b64(after)}" style="position:absolute;inset:0;width:100%;height:100%">
      <div id="b" style="position:absolute;inset:0;width:50%;overflow:hidden">
        <img src="data:image/jpeg;base64,{b64(before)}" style="height:100%;width:auto;max-width:none;display:block"
             id="bi"></div>
      <div id="l" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 4px #000"></div>
      <span style="position:absolute;left:8px;top:8px;color:#fff;background:#0008;padding:2px 6px;border-radius:4px;font:12px sans-serif">before</span>
      <span style="position:absolute;right:8px;top:8px;color:#fff;background:#0008;padding:2px 6px;border-radius:4px;font:12px sans-serif">after</span>
    </div>
    <script>
      const c=document.getElementById('c'),b=document.getElementById('b'),l=document.getElementById('l'),bi=document.getElementById('bi');
      function fit(){{bi.style.width=c.clientWidth+'px';}} fit(); window.addEventListener('resize',fit);
      function mv(x){{const r=c.getBoundingClientRect();let p=Math.min(Math.max((x-r.left)/r.width,0),1);
        b.style.width=(p*100)+'%';l.style.left=(p*100)+'%';}}
      let d=false;c.addEventListener('mousedown',e=>{{d=true;mv(e.clientX)}});
      window.addEventListener('mouseup',()=>d=false);c.addEventListener('mousemove',e=>{{if(d)mv(e.clientX)}});
      c.addEventListener('touchmove',e=>mv(e.touches[0].clientX));
    </script>"""
    components.html(html, height=disp_h + 10)  # streamlit==1.67 (pinned); st.iframe in newer versions


def load_upload(f):
    data = np.frombuffer(f.read(), np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError("Could not decode image")
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio("Enhancement mode", ["auto", "hybrid", "classical", "dl"], index=0,
                        help="auto = analyse the image, run all three and select the best by a no-reference score")
adaptive = st.sidebar.checkbox("Adaptive parameters (from image analysis)", value=True,
                               help="Untick to use fixed literature-default parameters (ablation)")
dev_choice = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
device = str(get_device(dev_choice))
nr_metrics = st.sidebar.checkbox("Compute NIQE / BRISQUE", value=True)
st.sidebar.caption(f"Running on **{device.upper()}**")

samples_dir = os.path.join(ROOT, "data", "real_unseen")
samples = sorted(os.listdir(samples_dir)) if os.path.isdir(samples_dir) else []

# ----------------------------------------------------------------------------- main
st.title("Robust Low-Light Image Restoration & Enhancement")
st.caption("Illumination-adaptive classical + Zero-DCE++ hybrid  ·  Problem P1")

c1, c2 = st.columns([2, 1])
with c1:
    up = st.file_uploader("Upload a low-light image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
with c2:
    gt_up = st.file_uploader("Optional: ground-truth (for PSNR/SSIM)", type=["png", "jpg", "jpeg", "bmp"])
    sample = st.selectbox("...or pick a sample", ["-"] + samples) if samples else "-"

img = None
if up is not None:
    img = load_upload(up)
elif sample != "-":
    img = E.imread(os.path.join(samples_dir, sample))

if img is None:
    st.info("Upload an image or choose a sample to start.")
    st.stop()

model = get_model(device)
with st.spinner("Enhancing..."):
    t0 = time.perf_counter()
    out, info = enhance(img, mode=mode, adaptive=adaptive, device=device, model=model)
    latency = (time.perf_counter() - t0) * 1000

x_in, was_gray = A.to_float_rgb(img)
stats, params = info["stats"], info["params"]

# --- condition report
st.subheader("1 · Detected image condition")
cc1, cc2 = st.columns([1, 1])
with cc1:
    for line in A.describe(stats, params):
        st.markdown("- " + line)
    if info.get("note"):
        st.warning(info["note"])
    st.markdown(f"- Mode: **{mode}** → used **{info.get('chosen', mode)}**")
    if "auto_scores" in info:
        st.markdown("- Auto-selection scores (higher = better): " +
                    ", ".join(f"{k} {v:.3f}" for k, v in info["auto_scores"].items()))
with cc2:
    with st.expander("All adaptive parameters"):
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()})
    with st.expander("Raw statistics"):
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.to_dict().items()})

# --- before / after
st.subheader("2 · Before / after")
before_after_slider(to_u8(x_in), to_u8(out))

buf = io.BytesIO()
Image.fromarray(to_u8(out)).save(buf, format="PNG")
name = (os.path.splitext(up.name)[0] if up is not None else os.path.splitext(sample)[0]) + f"_{info.get('chosen', mode)}.png"
st.download_button("Download enhanced image", buf.getvalue(), file_name=name, mime="image/png")

# --- intermediates
st.subheader("3 · Intermediate representations")
inter = info.get("intermediates", {})


def g(*keys):
    for k in keys:
        if k in inter and isinstance(inter[k], np.ndarray):
            return inter[k]
    return None


tiles = []
T = g("classical_illumination", "illumination")
if T is not None:
    tiles.append(("Illumination map T (guided-filter refined)", colormap(T)))
T0 = g("classical_illumination_init", "illumination_init")
if T0 is not None:
    tiles.append(("Initial max-RGB illumination", colormap(T0)))
R = g("classical_reflectance", "reflectance")
if R is not None:
    tiles.append(("Reflectance R = I / T", to_u8(R)))
for k, lab in [(("classical_relit", "relit"), "Retinex relit  I / T^γ"),
               (("classical_msrcr", "msrcr"), "MSRCR"),
               (("classical_denoised", "denoised"), "Guided-filter denoised"),
               (("classical_clahe", "clahe"), "Adaptive CLAHE (LAB L)"),
               (("classical_final",), "Classical output"),
               (("dl_prior_output",), "Zero-DCE++ (illumination-prior guided)"),
               (("dl_denoised",), "DL output after classical denoising"),
               (("fused",), "Pyramid fusion result")]:
    v = g(*k)
    if v is not None:
        tiles.append((lab, to_u8(v)))
cm = g("dl_curve_map", "curve_map")
if cm is not None:
    tiles.append(("Zero-DCE++ curve map |A|", colormap(np.abs(cm).mean(axis=2), cv2.COLORMAP_VIRIDIS)))
pm = g("prior_modulation")
if pm is not None:
    tiles.append(("Illumination prior m (curve strength)", colormap(pm, cv2.COLORMAP_VIRIDIS)))
fw = inter.get("fusion_weights")
if isinstance(fw, dict) and fw:
    names = list(fw)
    rgb = np.zeros(fw[names[0]].shape + (3,), np.float32)
    for i, n in enumerate(names[:3]):
        rgb[..., [2, 0, 1][i]] = fw[n]
    tiles.append((f"Fusion weights (blue={names[0]}, red={names[1] if len(names) > 1 else '-'})", to_u8(rgb)))
tiles.append(("Final output", to_u8(out)))

ncol = 4
for i in range(0, len(tiles), ncol):
    cols = st.columns(ncol)
    for col, (lab, im) in zip(cols, tiles[i:i + ncol]):
        col.image(im, caption=lab, width="stretch")

if "dl_strength_table" in inter or "strength_table" in inter:
    tab = inter.get("dl_strength_table", inter.get("strength_table"))
    st.caption("Quality-aware curve strength selection (k, score): " +
               ", ".join(f"{k:.1f}: {s:.3f}" for k, s in tab) +
               f"  →  chosen k = {inter.get('dl_strength', inter.get('strength'))}")

# --- metrics
st.subheader("4 · Metrics")
iqa = get_iqa()
s_in, s_out, amp = E.noise_stats(out, x_in)
m = {
    "Latency (ms)": f"{latency:.0f}",
    "Noise σ before → after (/255)": f"{s_in * 255:.2f} → {s_out * 255:.2f}",
    "Noise amplification (↓)": f"{amp:.2f}",
    "Newly clipped pixels %": f"{E.clipped_pct(out, x_in):.2f}",
    "Colour shift ΔE00 vs input chroma": f"{E.colour_shift(out, x_in):.2f}",
    "Halo score (1 = none)": f"{E.halo_score(out, x_in):.2f}",
    "Edge preservation (EPI vs input)": f"{E.edge_preservation_index(out, x_in):.3f}",
}
if nr_metrics:
    m["NIQE in → out (↓)"] = f"{iqa.nr('niqe', x_in):.2f} → {iqa.nr('niqe', out):.2f}"
    m["BRISQUE in → out (↓)"] = f"{iqa.nr('brisque', x_in):.2f} → {iqa.nr('brisque', out):.2f}"
if gt_up is not None:
    gt = A.to_float_rgb(load_upload(gt_up))[0]
    if gt.shape != out.shape:
        gt = cv2.resize(gt, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA)
    m["PSNR (dB)"] = f"{E.psnr(out, gt):.2f}"
    m["SSIM"] = f"{E.ssim(out, gt):.3f}"
    lp = iqa.lpips(out, gt)
    if np.isfinite(lp):
        m["LPIPS"] = f"{lp:.3f}"
    m["ΔE00 vs GT"] = f"{E.delta_e(out, gt):.2f}"
cols = st.columns(4)
for i, (k, v) in enumerate(m.items()):
    cols[i % 4].metric(k, v)

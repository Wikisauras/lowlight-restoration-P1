"""Fetch ONLY the 15-pair LOL test split (eval15) from the HF zip using HTTP range
requests (downloads a few MB instead of 347 MB).  Output: data/LOL/eval15/{low,high}.
Run from the project root:  python scripts/get_lol_eval15.py
Needs:  pip install remotezip
"""
import os, sys
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip"
OUT = os.path.join("data", "LOL", "eval15")

with RemoteZip(URL) as z:
    names = [n for n in z.namelist() if not n.endswith("/")]
    print("entries in zip:", len(names))
    sel = [n for n in names if "eval15" in n.replace("\\", "/").split("/")]
    print("eval15 files:", len(sel))
    if not sel:
        print("No eval15 folder found. First entries:", names[:10]); sys.exit(1)
    for n in sel:
        parts = n.replace("\\", "/").split("/")
        rel = parts[parts.index("eval15") + 1:]          # low/xxx.png or high/xxx.png
        dst = os.path.join(OUT, *rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(z.read(n))
for sub in ("low", "high"):
    d = os.path.join(OUT, sub)
    print(sub, len(os.listdir(d)) if os.path.isdir(d) else "MISSING")

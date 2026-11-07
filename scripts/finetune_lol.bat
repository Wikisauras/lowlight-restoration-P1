@echo off
REM Optional: fine-tune Zero-DCE++ on LOL (about 20-30 min on an RTX 4060), then re-evaluate with the new weights.
python -m src.train --low data/LOL/our485/low --high data/LOL/our485/high --epochs 30 --bs 8
python -m src.evaluation --paired LOL=data/LOL/eval15 --out results_finetuned --weights weights/ZeroDCE++_LOL_finetuned.pth --latency_n 0

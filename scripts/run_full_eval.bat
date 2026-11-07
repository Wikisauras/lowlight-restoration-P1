@echo off
REM One command for the full evaluation on the real datasets (run from the project root, venv active).
REM Missing dataset folders are skipped automatically.
python -m src.evaluation ^
  --paired LOL=data/LOL/eval15 ^
  --unpaired data/unpaired ^
  --synthetic data/synthetic ^
  --out results_real ^
  --ablation --latency_n 10 --device auto

#!/usr/bin/env bash
set -euo pipefail

uv run python main.py --full-simulation --sim-config results/ral_ablation_case02_random_2026052501/configs/case02_three_cs_proposed_seed_2026052501.json --environment-mode random --obstacle-seed 2026052501 --source-config results/ral_ablation_case02_random_2026052501/sources/case02_three_cs_seed_2026052501.json --birth --max-sources 3 --adaptive-dwell --measurement-time-s 30 --output-tag case02_three_cs_proposed_seed_2026052501

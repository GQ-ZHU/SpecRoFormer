#!/bin/bash
# =============================================================================
# environment.sh
#
# One-shot installation script for the SpecRoFormer repository.
#
# It assumes a working Python (>= 3.9) with pip, and either:
#   - a conda environment is already activated, or
#   - a virtualenv / system Python is available.
#
# Usage:
#   bash environment.sh
#
# To install CUDA-enabled PyTorch, install torch manually first:
#   https://pytorch.org/get-started/locally/
# then run this script; pip will keep the existing torch.
# =============================================================================

set -e

PYTHON=${PYTHON:-python}

echo "============================================================"
echo "SpecRoFormer: environment setup"
echo "============================================================"

echo "[1/4] Upgrading pip ..."
$PYTHON -m pip install --upgrade pip

echo
echo "[2/4] Installing dependencies from requirements.txt ..."
$PYTHON -m pip install -r requirements.txt

echo
echo "[3/4] Installing SpecRoFormer in editable mode ..."
$PYTHON -m pip install -e .

echo
echo "[4/4] Running environment check ..."
$PYTHON scripts/check_env.py

echo
echo "============================================================"
echo "Done."
echo "============================================================"
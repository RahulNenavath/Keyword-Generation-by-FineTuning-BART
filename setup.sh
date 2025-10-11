#!/usr/bin/env bash
# ============================================
# setup.sh — Fast & Robust Conda Setup
# ============================================

# Use -e and pipefail; avoid -u because some conda hooks reference unset vars.
set -e -o pipefail

# -------- Config --------
ENV_NAME="keywordgen"
PYTHON_VERSION="3.12"
KERNEL_NAME="KeywordGen (Py3.12)"
# Choose channels family: "conda-forge" (default) or "defaults" or comma-separated list
CHANNELS="${CHANNELS:-conda-forge}"

echo "==> Environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "==> Channel family: $CHANNELS"

# -------- Conda presence & shell integration --------
if ! command -v conda &>/dev/null; then
  echo "❌ Conda not found in PATH. Please install Miniconda/Anaconda first."
  exit 1
fi

# Load conda shell functions (required for 'conda activate')
source "$(conda info --base)/etc/profile.d/conda.sh"

# -------- Accept ToS for Anaconda defaults (harmless if unused) --------
# Newer conda requires this before hitting repo.anaconda.com
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   || true

# -------- Speed up solves & enforce strict priority --------
echo "==> Configuring fast solver and channel priority..."
conda update -n base -y conda conda-libmamba-solver >/dev/null 2>&1 || true
conda config --set solver libmamba
conda config --set channel_priority strict

# Configure one channel family to avoid slow/ambiguous solves
conda config --remove-key channels >/dev/null 2>&1 || true
IFS=',' read -ra CHS <<<"$CHANNELS"
for ch in "${CHS[@]}"; do
  conda config --add channels "$ch"
done

# Optional hygiene
conda clean -a -y >/dev/null 2>&1 || true

# -------- Create/refresh environment --------
echo "==> Creating Conda environment: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true

if [[ -f "environment.yml" ]]; then
  # Use the YAML as-is (do NOT pass --override-channels; not supported here)
  conda env create -n "$ENV_NAME" -f environment.yml -y
  # Ensure requested Python version if YAML omitted it
  conda install -n "$ENV_NAME" -y "python=${PYTHON_VERSION}"
else
  conda create -n "$ENV_NAME" -y "python=${PYTHON_VERSION}"
fi

# -------- Activate (avoid set -u around conda hooks) --------
echo "==> Activating environment..."
conda activate "$ENV_NAME"

# If you prefer OpenBLAS and want to avoid MKL deactivate hook entirely, uncomment:
# conda install -y "blas=*=openblas"

# -------- Pip step (separate for speed/stability) --------
if [[ -f "requirements.txt" ]]; then
  echo "==> Installing pip packages..."
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r requirements.txt
fi

# -------- Jupyter kernel --------
echo "==> Registering Jupyter kernel..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$KERNEL_NAME"

echo "✅ Setup complete."
echo "➡  To use:  conda activate $ENV_NAME"
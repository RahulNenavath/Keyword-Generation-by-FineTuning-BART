#!/usr/bin/env bash
# ============================================
# setup.sh — Environment Setup Script
# ============================================

# Exit on any error
set -e

ENV_NAME="keywordgen"
PYTHON_VERSION="3.12"

echo "Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda env remove -n $ENV_NAME -y >/dev/null 2>&1 || true
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing dependencies from environment.yml..."
conda env update -f environment.yml --prune

echo "Upgrading pip & installing pip-only packages..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "KeywordGen (Py3.12)"

echo "Setup complete!"
echo "To start using your environment:"
echo "conda activate $ENV_NAME"
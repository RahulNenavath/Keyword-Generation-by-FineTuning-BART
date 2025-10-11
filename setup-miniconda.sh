#!/usr/bin/env bash
set -e  # Exit on error

# --- Detect OS ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MINICONDA_OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MINICONDA_OS="MacOSX"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# --- Download Miniconda Installer ---
MINICONDA_VERSION="latest"
INSTALLER="Miniconda3-${MINICONDA_VERSION}-${MINICONDA_OS}-x86_64.sh"
DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

echo "Downloading Miniconda installer for ${MINICONDA_OS}..."
wget -q "${DOWNLOAD_URL}" -O "${INSTALLER}" || curl -sS -o "${INSTALLER}" "${DOWNLOAD_URL}"

# --- Install Miniconda silently ---
INSTALL_PATH="$HOME/miniconda3"
echo "Installing Miniconda to ${INSTALL_PATH} ..."
bash "${INSTALLER}" -b -p "${INSTALL_PATH}"

# --- Initialize Conda for current shell ---
echo "Initializing conda ..."
"${INSTALL_PATH}/bin/conda" init bash >/dev/null 2>&1 || true
"${INSTALL_PATH}/bin/conda" init zsh  >/dev/null 2>&1 || true

# --- Cleanup ---
rm -f "${INSTALLER}"

echo "✅ Miniconda installation complete!"
echo "Please restart your shell or run: source ~/.bashrc"
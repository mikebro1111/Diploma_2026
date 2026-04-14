#!/bin/bash
# setup_envs_macos.sh
# Automates the installation of GIL and Free-threaded Python 3.13/3.14 via pyenv
# and sets up the required virtual environments.

set -e

# --- Configuration ---
PY313_VERSION="3.13.0"
PY313T_VERSION="3.13.0t"
PY314_VERSION="3.14-dev"
PY314T_VERSION="3.14-dev" # Free-threaded is usually built into the dev branch or specific suffix

# COLORS
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Free-threaded Python Benchmark — macOS/Linux Environment Setup"
echo "======================================================================"

# 1. Check for pyenv
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}Error: pyenv is not installed.${NC}"
    echo "Please install it first: brew install pyenv"
    exit 1
fi

echo -e "${GREEN}>>> Installing Python via pyenv...${NC}"

# Install 3.13
pyenv install $PY313_VERSION --skip-existing
pyenv install $PY313T_VERSION --skip-existing

# Install 3.14 (if possible)
# Note: 3.14 is currently in development.
pyenv install $PY314_VERSION --skip-existing

echo -e "${GREEN}>>> Creating Virtual Environments...${NC}"

# Python 3.13 GIL
$(pyenv prefix $PY313_VERSION)/bin/python -m venv venv_gil
source venv_gil/bin/activate
pip install --upgrade pip
pip install -r requirements_gil.txt
deactivate

# Python 3.13 No-GIL
$(pyenv prefix $PY313T_VERSION)/bin/python -m venv venv_nogil
source venv_nogil/bin/activate
pip install --upgrade pip
pip install -r requirements_nogil.txt
deactivate

# Python 3.14 GIL
$(pyenv prefix $PY314_VERSION)/bin/python -m venv venv_gil_314
source venv_gil_314/bin/activate
pip install --upgrade pip
pip install -r requirements_gil.txt
deactivate

# Python 3.14 No-GIL
# For 3.14, the free-threading build might require a specific environment variable or prefix
# In pyenv, 3.14-dev often includes the build option.
$(pyenv prefix $PY314_VERSION)/bin/python -m venv venv_nogil_314
source venv_nogil_314/bin/activate
pip install --upgrade pip
pip install -r requirements_nogil.txt
deactivate

echo -e "======================================================================"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "Environments ready:"
echo "  - venv_gil (3.13 GIL)"
echo "  - venv_nogil (3.13 Free-threaded)"
echo "  - venv_gil_314 (3.14 GIL)"
echo "  - venv_nogil_314 (3.14 Free-threaded)"
echo "======================================================================"

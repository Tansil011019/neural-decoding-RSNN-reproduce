#!/bin/bash

# Stop script on error
set -e

echo "Starting Setting Up"

VENV_DIR="venvs/efficient-rsnn-bmi"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "No virtual environment found, creating one..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
python3 -m pip install --require-virtualenv -e "."

echo "Project is completely set"
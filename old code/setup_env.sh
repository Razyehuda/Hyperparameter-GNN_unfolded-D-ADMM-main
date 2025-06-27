#!/bin/bash

echo "Setting up virtual environment for Deep-Unfolded-D-ADMM project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "    deactivate"
echo "" 
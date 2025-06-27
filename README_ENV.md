# Environment Setup for Deep-Unfolded-D-ADMM

This document provides instructions for setting up the virtual environment for the Deep-Unfolded-D-ADMM project.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Quick Setup

### Windows
1. Open Command Prompt or PowerShell in the project directory
2. Run the setup script:
   ```cmd
   setup_env.bat
   ```

### Linux/macOS
1. Open Terminal in the project directory
2. Make the script executable and run it:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

## Manual Setup

If you prefer to set up the environment manually:

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv

# Linux/macOS
python3 -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Required Packages

The following packages will be installed:

- **torch** (>=1.10.2): PyTorch deep learning framework
- **torchvision** (>=0.11.3): Computer vision utilities for PyTorch
- **numpy** (>=1.21.0): Numerical computing library
- **scipy** (>=1.7.0): Scientific computing library
- **tqdm** (>=4.62.0): Progress bar library
- **matplotlib** (>=3.5.0): Plotting library
- **networkx** (>=2.6.0): Graph theory and network analysis
- **tensorboardX** (>=2.4): TensorBoard logging for PyTorch
- **scikit-learn** (>=1.0.0): Machine learning utilities

## Using the Environment

### Activate Environment
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Deactivate Environment
```bash
deactivate
```

### Running the Code
After activating the environment, you can run the training scripts:

```bash
# For distributed LASSO
python dlasso.py --exp_name dlasso_with_50_agents --data simulated --batch_size 100 --P 50 --graph_prob 0.12 --case dlasso --model diff --valid True

# For distributed linear regression
python dlr.py --exp_name dlr_with_50_agents --data simulated --batch_size 100 --P 50 --graph_prob 0.12 --case dlr --model diff --valid True
```

## Troubleshooting

### CUDA Support
If you have a CUDA-capable GPU and want to use it:

1. Install the CUDA version of PyTorch:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Common Issues

1. **Permission Denied**: Make sure you have write permissions in the project directory
2. **Python Not Found**: Ensure Python is installed and added to your system PATH
3. **Package Installation Errors**: Try upgrading pip first: `pip install --upgrade pip`

## Notes

- The project was tested with Python 3.9.7, PyTorch 1.10.2, and CUDA 11.1
- Make sure to activate the virtual environment before running any scripts
- The environment includes all necessary dependencies for both distributed LASSO and linear regression problems 
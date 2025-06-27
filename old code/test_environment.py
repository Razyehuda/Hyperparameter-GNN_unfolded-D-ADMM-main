#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
and the project can be imported correctly.
"""

import sys

def test_imports():
    """Test all required imports for the Deep-Unfolded-D-ADMM project."""
    
    print("Testing imports for Deep-Unfolded-D-ADMM project...")
    print("=" * 50)
    
    # Test standard library imports
    try:
        import os
        import sys
        import time
        import gc
        print("✓ Standard library imports: OK")
    except ImportError as e:
        print(f"✗ Standard library imports: FAILED - {e}")
        return False
    
    # Test PyTorch and related imports
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
        print(f"✓ PyTorch imports: OK (PyTorch version: {torch.__version__})")
    except ImportError as e:
        print(f"✗ PyTorch imports: FAILED - {e}")
        return False
    
    # Test scientific computing imports
    try:
        import numpy as np
        import scipy
        print(f"✓ Scientific computing imports: OK (NumPy version: {np.__version__}, SciPy version: {scipy.__version__})")
    except ImportError as e:
        print(f"✗ Scientific computing imports: FAILED - {e}")
        return False
    
    # Test visualization imports
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imports: OK")
    except ImportError as e:
        print(f"✗ Matplotlib imports: FAILED - {e}")
        return False
    
    # Test graph theory imports
    try:
        import networkx as nx
        print("✓ NetworkX imports: OK")
    except ImportError as e:
        print(f"✗ NetworkX imports: FAILED - {e}")
        return False
    
    # Test utility imports
    try:
        from tqdm import tqdm
        from tensorboardX import SummaryWriter
        print("✓ Utility imports (tqdm, tensorboardX): OK")
    except ImportError as e:
        print(f"✗ Utility imports: FAILED - {e}")
        return False
    
    # Test scikit-learn imports
    try:
        import sklearn
        print("✓ Scikit-learn imports: OK")
    except ImportError as e:
        print(f"✗ Scikit-learn imports: FAILED - {e}")
        return False
    
    # Test project-specific imports
    try:
        import configurations
        import utils
        import models
        import DADMM_utils
        import LoadData
        print("✓ Project-specific imports: OK")
    except ImportError as e:
        print(f"✗ Project-specific imports: FAILED - {e}")
        return False
    
    print("=" * 50)
    print("All imports successful! Environment is ready for use.")
    return True

def test_basic_functionality():
    """Test basic functionality of key components."""
    
    print("\nTesting basic functionality...")
    print("=" * 50)
    
    try:
        # Test PyTorch device detection
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device detection: {device}")
        
        # Test tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("✓ Tensor operations: OK")
        
        # Test NumPy operations
        import numpy as np
        a = np.random.randn(3, 3)
        b = np.random.randn(3, 3)
        c = np.dot(a, b)
        print("✓ NumPy operations: OK")
        
        # Test NetworkX graph creation
        import networkx as nx
        G = nx.erdos_renyi_graph(10, 0.3)
        print(f"✓ NetworkX graph creation: OK (Graph with {G.number_of_nodes()} nodes)")
        
        print("=" * 50)
        print("All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Deep-Unfolded-D-ADMM Environment Test")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_imports()
    functionality_ok = False
    if imports_ok:
        functionality_ok = test_basic_functionality()
    
    if imports_ok and functionality_ok:
        print("\n🎉 SUCCESS: Environment is fully configured and ready to use!")
        print("\nYou can now run the training scripts:")
        print("  python dlasso.py --exp_name test --data simulated --batch_size 10 --P 5 --graph_prob 0.5 --case dlasso --model diff --valid True")
        print("  python dlr.py --exp_name test --data simulated --batch_size 10 --P 5 --graph_prob 0.5 --case dlr --model diff --valid True")
    else:
        print("\n❌ FAILURE: Environment setup incomplete. Please check the errors above.")
        sys.exit(1) 
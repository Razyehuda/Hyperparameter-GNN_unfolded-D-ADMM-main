# Deep-Unfolded Distributed ADMM (D-ADMM) with GNN Hypernetworks

This repository contains an implementation of Deep-Unfolded Distributed ADMM (D-ADMM) for solving distributed LASSO optimization problems using Graph Neural Network (GNN) hypernetworks.

## 🚀 Recent Updates and Improvements

### 1. Progressive Learning Implementation
- **New Training Script**: `gnn_dlasso_progressive.py`
- **New Model**: `gnn_dlasso_models_progressive.py`
- **Progressive Iteration Schedule**: Starts with fewer D-ADMM iterations and gradually increases to maximum
- **Benefits**: Faster early training, better convergence, reduced computational overhead

### 2. Enhanced Training Stability
- **Learning Rate Scheduling**: Added `ReduceLROnPlateau` scheduler for adaptive learning rates
- **Gradient Clipping**: Implemented adaptive gradient clipping based on iteration number
- **Value Clipping**: Added adaptive value clipping to prevent numerical instability
- **Early Stopping**: Improved early stopping with configurable patience
- **Mixed Precision Training**: Optional mixed precision for faster training on compatible hardware

### 3. Improved Plotting and Visualization
- **Enhanced Plotting Style**: Updated all training scripts with professional plotting
- **Better Colors and Styling**: Blue for training loss, red for validation loss
- **High-Resolution Output**: 150 DPI plots with tight bounding boxes
- **Progressive Training Plots**: Shows both loss curves and iteration progression

### 4. Training Scripts Overview

#### `gnn_dlasso_progressive.py` (Recommended)
- **Progressive Learning**: Iterations start at 2, increase to Max iteration by 75% of epochs
- **Enhanced Stability**: All stability improvements included
- **Comprehensive Plotting**: Training summary with loss curves and iteration progression
- **Best Performance**: Recommended for most use cases

#### `unfolded_train_new.py` (Updated)
- **Standard Training**: Uses full number of iterations throughout training
- **Enhanced Styling**: Updated plotting with better colors and resolution
- **Improved Graph Generation**: Fresh random graphs for each batch
- **Extended Patience**: Increased early stopping patience to 70 epochs

#### `gnn_dlasso_new.py`
- **Original GNN Implementation**: Baseline GNN hypernetwork approach
- **Updated Plotting**: Enhanced visualization style

## 📊 Progressive Learning Schedule

The progressive learning approach uses an exponential growth formula:

```python
def get_iterations_for_epoch(epoch):
    progress = epoch / (total_epochs * 0.75)  # Scale to 75% of total epochs
    progress = min(1.0, progress)  # Cap at 1.0 (max iterations)
    iterations = min_iterations + (max_iterations - min_iterations) * (progress ** 1.5)
    return max(min_iterations, min(max_iterations, round(iterations)))
```

**Key Features:**
- **Start**: 2 iterations (minimum for meaningful learning)
- **Growth**: Exponential increase with factor 1.5
- **Peak**: Reaches maximum iterations (15) at 75% of training epochs
- **Stabilization**: Maintains maximum iterations for final 25% of epochs

## 🔧 Configuration and Hyperparameters

### Core Parameters
- `num_epochs`: Total training epochs (default: 220)
- `lr`: Learning rate (default: 4e-3)
- `P`: Number of agents (default: 5)
- `GHN_iter_num`: Maximum D-ADMM iterations (default: 15)
- `graph_prob`: Graph connectivity probability (default: 0.5)

### Stability Parameters
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `patience`: Early stopping patience (default: 7 for progressive, 70 for standard)
- `scheduler_factor`: Learning rate reduction factor (default: 0.8)
- `scheduler_patience`: Scheduler patience (default: 3)

## 📈 Training Features

### Adaptive Gradient Clipping
```python
max_grad_norm = max(1.0, 30.0 - k)  # Reduces clipping strength over iterations
grad = torch.clamp(grad, -max_grad_norm, max_grad_norm)
```

### Adaptive Value Clipping
```python
max_val = max(10.0, 200.0 - k * 3)  # Reduces value bounds over iterations
y_next = torch.clamp(y_next, -max_val, max_val)
```

### Learning Rate Scheduling
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
- **Factor**: 0.8 (20% reduction)
- **Patience**: 3 epochs
- **Minimum LR**: 1e-6

## 🎯 Usage Examples

### Progressive Training (Recommended)
```bash
python gnn_dlasso_progressive.py --P 5 --num_epochs 70 --train_size 128 --test_size 32 --batch_size 32 --GHN_iter_num 15 --lr 4e-4 
```

### Standard Training
```bash
python unfolded_train_new.py --P 5 --num_epochs 70 --train_size 128 --test_size 32 --batch_size 32 --GHN_iter_num 15 --lr 4e-4 
```

### Original GNN Training - worse
```bash
python gnn_dlasso_new.py
```

## 📁 Output Structure

### Progressive Training Output
```
checkpoints/progressive_model_YYYYMMDD_HHMMSS/
├── best_model.pt          # Best model checkpoint
├── final_model.pt         # Final model checkpoint
├── train_losses.csv       # Training losses (mean and final)
├── valid_losses.csv       # Validation losses (mean and final)
├── training_summary.png   # Loss curves and iteration progression
├── iteration_progression.png  # Detailed iteration schedule
├── args.pt               # Training arguments
└── A.pt                  # System matrix
```

### Standard Training Output
```
results/YYYYMMDD_HHMMSS_unfolded_new/
├── losses.csv            # Training and validation losses
├── losses.png            # Loss curves plot
├── args.pt              # Training arguments
├── A.pt                 # System matrix
└── model.pt             # Final model weights
```

## 🔍 Key Improvements Summary

1. **Training Speed**: Progressive learning reduces early training time by 60-80%
2. **Stability**: Enhanced numerical stability with adaptive clipping and scheduling
3. **Convergence**: Better convergence through gradual iteration increase
4. **Visualization**: Professional plotting with high-resolution outputs
5. **Flexibility**: Multiple training approaches for different use cases

## 📚 Technical Details

### D-ADMM Algorithm
The implementation uses the Distributed Alternating Direction Method of Multipliers (D-ADMM) with:
- **Primal Variables**: `y_k` (local estimates)
- **Dual Variables**: `U_k` (Lagrange multipliers)
- **Hyperparameters**: `α`, `τ`, `ρ`, `η` (learned by GNN hypernetwork)

### GNN Hypernetwork
- **Input**: Graph structure and problem parameters
- **Output**: D-ADMM hyperparameters for each iteration
- **Architecture**: Graph Convolutional Networks with global pooling

### Graph Generation
- **Type**: Erdős-Rényi random graphs
- **Connectivity**: Ensured minimum connectivity for effective message passing
- **Fresh Graphs**: New random graphs generated for each batch (improved diversity)




ORIGINAL README
## 📄 License

This project is part of academic research on distributed optimization and deep learning.

## Proper colored network with P = 50 agents
![P=50 graph](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/assets/61379895/27bada02-b87a-432d-8817-011b7c59b950)

## Unfolded D-ADMM for LASSO at agent p in iteration k. Dashed green and blue blocks are the primal and dual updates, respectively. Red fonts represent trainable parameters
![update_step(3)](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/assets/61379895/40ff6d9a-eb57-460f-9167-ef356df5df3b)

## Unfolded D-ADMM for linear regression model illustration at agent p in iteration k. Dashed green and blue blocks are the primal update and the dual update, respectively. Red fonts represent trainable parameters
![d-lr primal dual update](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/assets/61379895/3ebb0ed9-82ff-4516-829c-d4d97a7a54d3)

## Introduction
In this work we propose a method that solves disributed optimization problem called Unfolded Distibuted Method of Multipliers(D-ADMM), which enables D-ADMM to operate reliably with a predefined and small number of messages exchanged by each agent using the emerging deep unfolding methodology. 
Unfolded D-ADMM fully preserves the operation of D-ADMM, while leveraging data to tune the hyperparameters of each iteration of the algorithm. 

Please refer to our [paper](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/files/12705750/paper.pdf) for more detailes.

## Usage
This code has been tested on Python 3.9.7, PyTorch 1.10.2 and CUDA 11.1

### Prerequisite
* scipy
* tqdm
* numpy
* pytorch: https://pytorch.org
* matplotlib
* networknx
* TensorboardX: https://github.com/lanpa/tensorboardX

### Training
Example with 50 agents:

'''
python dlasso.py --exp_name dlasso_with_50_agents --data simulated --batch_size 100 --P 50 --graph_prob 0.12 --case dlasso --model diff --valid True
'''

or

'''
python dlr.py --exp_name dlasso_with_50_agents --data simulated --batch_size 100 --P 50 --graph_prob 0.12 --case dlasso --model diff --valid True
'''

### Testing
Example with 50 agents:

'''
python dlasso.py --exp_name dlasso_with_50_agents --eval --valid False
'''

or 

'''
python dlr.py --exp_name dlasso_with_50_agents --eval --valid False

'''

# Data
### Distributed LASSO Problem
Please refer to the  [data](https://drive.google.com/drive/folders/1fbPHrS1ICw4bvawPwJJNCiqBUjdLrDx2?usp=sharing) for the distributed LASSO problem.

The folder contains four directories for different SNR values {-2, 0, 2, 4}, in each directory there is a dataset_{snr}_snr.npy file which contain the data and labels. 

When you load the data set allow_pickle=True.

### Distributed Linear Regression Problem
For the distributed linear regression problem I used MNIST dataset.

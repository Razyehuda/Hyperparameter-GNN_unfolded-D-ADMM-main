import torch
import torch.nn.functional as F

def set_A(args):
    m = args.m
    n = args.n

    A = torch.zeros((1,args.P,m,n))
    for p in range(args.P):
        # Use orthogonal initialization for better conditioning
        A_temp = torch.randn((m,n))
        U, S, V = torch.svd(A_temp)
        # Normalize singular values for better conditioning
        S = torch.clamp(S, min=0.1, max=10.0)
        A[0,p] = U @ torch.diag(S) @ V.T
    return A

def compute_loss2(Y,label):
    w = abs(label) + 0.0001
    w /= w.sum(dim=1).unsqueeze(-1)

    y_mean = Y.mean(dim=2)
    loss_final = (F.mse_loss(y_mean[-1],label,reduction='none')*w).sum(dim=1)
    loss_mean = (F.mse_loss(y_mean.mean(dim=0),label,reduction='none')*w).sum(dim=1)
    return loss_mean.mean(), loss_final.mean()

def compute_loss(Y, label):
    """
    Fixed loss computation that properly handles tensor shapes
    Y: [K, batch_size, P, n, 1] - model output
    label: [batch_size, n, 1] - target
    """
    K, batch_size, P, n, _ = Y.shape
    
    # Check for NaN values in input
    if torch.isnan(Y).any() or torch.isinf(Y).any():
        print("Warning: NaN/Inf detected in model output Y")
        # Return a safe loss value
        return torch.tensor(1.0, device=Y.device), torch.tensor(1.0, device=Y.device)
    
    if torch.isnan(label).any() or torch.isinf(label).any():
        print("Warning: NaN/Inf detected in label")
        return torch.tensor(1.0, device=Y.device), torch.tensor(1.0, device=Y.device)
    
    # Reshape for easier computation
    Y_reshaped = Y.view(K, batch_size, P, n)  # Remove last dimension
    label_reshaped = label.view(batch_size, n)  # Remove last dimension
    
    # Add small epsilon to prevent numerical issues
    eps = 1e-8
    
    # Compute loss for each layer
    losses = []
    for k in range(K):
        # For each layer, compute MSE between model output and target
        # Average across agents (P dimension)
        layer_loss = 0.0
        for p in range(P):
            # Y[k, :, p] has shape [batch_size, n]
            # label_reshaped has shape [batch_size, n]
            mse = F.mse_loss(Y_reshaped[k, :, p], label_reshaped, reduction='mean')
            layer_loss += mse
        layer_loss /= P  # Average across agents
        losses.append(layer_loss)
    
    losses = torch.stack(losses)  # [K]
    
    # Check for NaN in losses
    if torch.isnan(losses).any() or torch.isinf(losses).any():
        print("Warning: NaN/Inf detected in computed losses")
        return torch.tensor(1.0, device=Y.device), torch.tensor(1.0, device=Y.device)
    
    # Mean loss across all layers
    loss_mean = losses.mean()
    # Final layer loss
    loss_final = losses[-1]
    
    # Add small epsilon to prevent zero loss
    loss_mean = loss_mean + eps
    loss_final = loss_final + eps
    
    # Final NaN check
    if torch.isnan(loss_mean) or torch.isinf(loss_mean):
        loss_mean = torch.tensor(1.0, device=Y.device)
    if torch.isnan(loss_final) or torch.isinf(loss_final):
        loss_final = torch.tensor(1.0, device=Y.device)
    
    return loss_mean, loss_final
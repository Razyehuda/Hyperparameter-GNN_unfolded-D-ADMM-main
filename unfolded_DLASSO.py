import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import math


class DLASSO_unfolded(nn.Module):
    def __init__(self, A, args):
        super().__init__()
        # A [1, P, m, d]
        self.A = A
        _, self.P, self.m, self.n = self.A.shape

        self.AtA = self.compute_Atx(self.A)

        self.K = args.GHN_iter_num

        self.DADMM_mode = args.DADMM_mode
        if args.DADMM_mode == 'same':
            hyp_shape = [self.K,1,4]
        else:
            hyp_shape = [self.K,self.P,4]

        max_param = torch.tensor([args.alpha_max, args.tau_max, args.rho_max, args.eta_max], device=A.device)
        self.seq_hyp = seq_hyperparam(hyp_shape,max_param, args)

        self.max_param = max_param.unsqueeze(0)
        
        # Store args for configurable parameters
        self.args = args

    def forward(self, b, graph_list,K=None):
        # Inputs
        # b: [batch_size,m,P]
        # graph_list: list of len batch_size, each item is graph with P agents
        batch_size = max(len(b), len(graph_list))
        device = b.device
        if K is None:
            K = self.K
        else:
            K = min(K,self.K)

        Atb = self.compute_Atx(b)
        sum_neighbors = self.compute_sum_neighbors(graph_list, device=device)

        Y = []
        y_k = torch.randn((batch_size, self.P, self.n, 1), device=device)*1e-2  # [batch_size,P,d,1]
        U_k = torch.randn((batch_size, self.P, self.n, 1), device=device)*1e-2  # [batch_size,P,d,1]
        delta = torch.randn((batch_size, self.P, self.n, 1), device=device)*1e-2

        for k in range(K):
            # Check for NaN values and reset if necessary
            if torch.isnan(y_k).any() or torch.isinf(y_k).any():
                print(f"Warning: NaN/Inf detected in y_k at iteration {k}, resetting...")
                y_k = torch.zeros_like(y_k)
            
            if torch.isnan(U_k).any() or torch.isinf(U_k).any():
                print(f"Warning: NaN/Inf detected in U_k at iteration {k}, resetting...")
                U_k = torch.zeros_like(U_k)

            hyp = self.seq_hyp(k)
            alpha_k = hyp[:, 0].unsqueeze(0).unsqueeze(-1)  # [b,P,1,1]
            tau_k = hyp[:, 1].unsqueeze(0).unsqueeze(-1)  # [b,P,1,1]
            rho_k = hyp[:, 2].unsqueeze(0).unsqueeze(-1)  # [b,P,1,1]
            eta_k = hyp[:, 3].unsqueeze(0).unsqueeze(-1)   # [b,P,1,1]

            AtAy = torch.zeros((batch_size,self.P,self.n,1), device=device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(self.AtA[0,p], y_k[:, p])

            grad = (AtAy
                    - Atb
                    + y_k.sign() * tau_k
                    + U_k * sum_neighbors
                    + delta * rho_k)
            
            # Adaptive gradient clipping based on iteration
            max_grad_norm = max(1.0, 30.0 - k)  # Reduce clipping threshold over iterations
            grad = torch.clamp(grad, -max_grad_norm, max_grad_norm)
            
            # Check for NaN in gradient
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"Warning: NaN/Inf in gradient at iteration {k}, skipping update...")
                grad = torch.zeros_like(grad)
            
            # Update variables with stability checks
            y_next = y_k - alpha_k * grad
            
            # Adaptive value clipping based on iteration
            max_val = max(10.0, 200.0 - k * 3)  # Reduce max values over iterations
            y_next = torch.clamp(y_next, -max_val, max_val)

            delta = self.compute_delta(graph_list, y_next, device=device)
            #delta = torch.clamp(delta, -5.0, 5.0)  # Tighter delta bounds

            U_k = U_k + delta * eta_k
            U_k = torch.clamp(U_k, -max_val, max_val)
            
            # Final NaN check before updating
            if torch.isnan(y_next).any() or torch.isinf(y_next).any():
                print(f"Warning: NaN/Inf in y_next at iteration {k}, using previous value...")
                y_next = y_k
            
            y_k = y_next
            Y.append(y_k)

        Y = torch.stack(Y)
        return Y, hyp
    def compute_sum_neighbors(self,graph_list, device):
        batch_size = len(graph_list)
        sum_neighbors = torch.zeros((batch_size, self.P, 1, 1), device=device)  # [batch_size,P,1,1]
        for i in range(batch_size):
            graph = graph_list[i]
            for p in range(self.P):
                sum_neighbors[i, p] = len(list(graph.neighbors(p)))
        return sum_neighbors

    def compute_Atx(self,x):
        Atx = torch.zeros((x.shape[0],self.P,self.n,x.shape[3]), device=x.device)
        for p in range(self.P):
            Atx[:,p] = torch.matmul(self.A[0,p].T, x[:,p])
        return Atx


    def compute_delta(self, graph_list, y1, y2=None, device=None):
        if y2 is None:
            y2 = y1
        delta = torch.zeros_like(y1, device=device)  # [b,P,d,1]
        batch_size = len(graph_list)
        for b in range(batch_size):
            graph = graph_list[b]
            for p in range(self.P):
                y_p = y1[b, p]
                for j in graph.neighbors(p):
                    diff = y_p - y2[b, j]
                    delta[b, p] += diff
                    delta[b,j] -= diff
        return delta

    def compute_loss(self, y_k, label):
        loss = 0.0
        for p in range(self.P):
            loss += F.mse_loss(y_k[:,p],label)
        return loss/self.P

class seq_hyperparam(nn.Module):
    def __init__(self,hyp_shape,max_param, args=None):
        super().__init__()
        # Initialize all parameters to zero (bad start, but will allow learning)
        self.param = nn.Parameter(torch.zeros(hyp_shape))
        self.max_param = max_param.unsqueeze(0)
        self.args = args
    
    def forward(self,k):
        hyp = torch.sum(self.param[:k+1],dim=0).squeeze(0)
        hyp = torch.sigmoid(hyp) * self.max_param
        # Add regularization to encourage balanced hyperparameter values
        # Penalize when all parameters are close to their maximum values
        if self.training:
            # Add small regularization to prevent all parameters from reaching max
            max_penalty = torch.sum(hyp) / (hyp.shape[0] * hyp.shape[1])  # Average across batch and agents
            if self.args is not None and max_penalty > self.args.max_penalty_threshold:  # If average is too high
                hyp = hyp * self.args.penalty_reduction_factor  # Slightly reduce all parameters
        # Clamp hyperparameter output with tighter bounds for stability
        hyp = torch.clamp(hyp, min=1e-4, max=0.99)
        return hyp.unsqueeze(-1)

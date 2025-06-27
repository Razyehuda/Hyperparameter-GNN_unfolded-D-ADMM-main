import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import math


class GNNHypernetwork3(nn.Module):
    def __init__(self, P, m, hidden_dim):
        super(GNNHypernetwork3, self).__init__()
        self.P = P
        self.m = m

        # GNN layers with improved architecture
        self.conv1 = GCNConv(self.m, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, 4 * hidden_dim)
        self.conv4 = GCNConv(4 * hidden_dim, 4 * hidden_dim)
        self.conv5 = GCNConv(4 * hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(4 * hidden_dim)
        
        # Batch normalization for better training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)
        self.bn4 = nn.BatchNorm1d(4 * hidden_dim)
        self.bn5 = nn.BatchNorm1d(4 * hidden_dim)

        # Initialization
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                nn.init.zeros_(conv.lin.bias)

    def forward(self, x, graph_list):
        batch_size = x.shape[0]
        hyp = torch.stack([self.graph_conv(x[i], graph_list[i]) for i in range(batch_size)])
        return hyp.view(batch_size, -1)

    def graph_conv(self, x, nx_graph):
        edge_index = from_networkx(nx_graph).edge_index
        x = x.squeeze(-1)
        
        # Ensure both x and edge_index are on the same device
        device = x.device
        edge_index = edge_index.to(device)
        x = x.to(device)
        
        # Improved forward pass with batch norm
        x1 = F.leaky_relu(self.conv1(x, edge_index))
        x1 = self.bn1(x1)
        x1 = self.dropout(x1)
        
        x2 = F.leaky_relu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)
        x2 = self.dropout(x2)
        
        x3 = F.leaky_relu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)
        x3 = self.dropout(x3)
        
        x4 = F.leaky_relu(self.conv4(x3, edge_index))
        x4 = self.bn4(x4)
        x4 = self.dropout(x4)
        
        x5 = F.leaky_relu(self.conv5(x4, edge_index))
        x5 = self.bn5(x5)
        x5 = self.norm(x5)
        
        return x5.view(-1)


class DLASSO_GNNHyp3_Progressive(nn.Module):
    def __init__(self, A, args):
        super().__init__()
        # A [1, P, m, d]
        self.A = A
        _, self.P, self.m, self.n = self.A.shape

        self.AtA = self.compute_Atx(self.A)
        self.K = args.GHN_iter_num
        hidden_dim = args.GHyp_hidden
        self.DADMM_mode = args.DADMM_mode

        # Improved encoder
        self.encoder = GNNHypernetwork3(P=self.P,
                                       m=self.n * 2,
                                       hidden_dim=hidden_dim)

        # Improved decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(self.P * 4 * hidden_dim, 4 * hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(4 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU()
        )
        
        if args.DADMM_mode == 'same':
            self.fc = nn.Linear(hidden_dim, 4)
        else:
            self.fc = nn.Linear(hidden_dim, 4 * self.P)
        
        # Better initialization for progressive learning
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)
        
        # Initialize bias for progressive learning - start with conservative values
        with torch.no_grad():
            # Start with smaller hyperparameters for early iterations
            self.fc.bias.data[0] = -0.5  # alpha: sigmoid(-0.5) * 0.3 ≈ 0.11 (conservative)
            self.fc.bias.data[1] = -1.0  # tau: sigmoid(-1.0) * 0.99 ≈ 0.27 (moderate)
            self.fc.bias.data[2] = -0.8  # rho: sigmoid(-0.8) * 0.99 ≈ 0.31 (moderate)
            self.fc.bias.data[3] = -1.2  # eta: sigmoid(-1.2) * 0.99 ≈ 0.23 (conservative)

        # Parameter bounds - no scaling to allow full learning capacity
        self.alpha_max = torch.tensor(args.alpha_max)
        self.tau_max = torch.tensor(args.tau_max)
        self.rho_max = torch.tensor(args.rho_max)
        self.eta_max = torch.tensor(args.eta_max)

    def forward(self, b, graph_list, training_iterations=None):
        # Inputs
        # b: [batch_size,m*P]
        # graph_list: list of len batch_size, each item is graph with P agents
        # training_iterations: optional override for progressive learning
        batch_size = max(len(b), len(graph_list))
        K = training_iterations if training_iterations is not None else self.K

        Atb = self.compute_Atx(b)
        sum_neighbors = self.compute_sum_neighbors(graph_list)

        Y = []
        # Initialize state variables - use small random initialization like unfolded model
        y_k = torch.randn((batch_size, self.P, self.n, 1), device=b.device) * 1e-2
        U_k = torch.randn((batch_size, self.P, self.n, 1), device=b.device) * 1e-2
        delta = torch.randn((batch_size, self.P, self.n, 1), device=b.device) * 1e-2

        for k in range(K):
            # Check for NaN values and reset if necessary
            if torch.isnan(y_k).any() or torch.isinf(y_k).any():
                print(f"Warning: NaN/Inf detected in y_k at iteration {k}, resetting...")
                y_k = torch.zeros_like(y_k)
            
            if torch.isnan(U_k).any() or torch.isinf(U_k).any():
                print(f"Warning: NaN/Inf detected in U_k at iteration {k}, resetting...")
                U_k = torch.zeros_like(U_k)

            AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            # Ensure AtA is on the same device as b
            AtA_device = self.AtA.to(b.device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(AtA_device[0, p], y_k[:, p])
            
            # Generate hyperparameters
            hyp = torch.cat([AtAy, Atb], dim=2)
            hyp = self.encoder(hyp, graph_list)
            hyp = self.decoder(hyp)
            hyp = self.fc(hyp)
            hyp = torch.sigmoid(hyp)
            
            # Clamp hypernetwork output with tighter bounds for stability
            hyp = torch.clamp(hyp, min=1e-4, max=0.9999)

            if self.DADMM_mode == 'same':
                hyp = hyp.view(batch_size, 4, 1, 1, 1)
            else:
                hyp = hyp.view(batch_size, 4, self.P, 1, 1)

            # Extract parameters with adaptive bounds based on iteration
            # Ensure hyperparameter tensors are on the same device as hyp
            alpha_max_device = self.alpha_max.to(hyp.device)
            tau_max_device = self.tau_max.to(hyp.device)
            rho_max_device = self.rho_max.to(hyp.device)
            eta_max_device = self.eta_max.to(hyp.device)
            
            alpha_k = hyp[:, 0] * alpha_max_device
            tau_k = hyp[:, 1] * tau_max_device
            rho_k = hyp[:, 2] * rho_max_device
            eta_k = hyp[:, 3] * eta_max_device

            # No decay factor - let the model learn freely
            # Only apply reasonable bounds to prevent extreme values
            # alpha_k is already bounded by alpha_max = 0.3
            tau_k = torch.clamp(tau_k, max=0.9999)     # Almost no limit on tau
            rho_k = torch.clamp(rho_k, max=0.9999)     # Almost no limit on rho
            eta_k = torch.clamp(eta_k, max=0.9999)     # Almost no limit on eta

            # Compute gradient
            AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            # Ensure AtA is on the same device as b
            AtA_device = self.AtA.to(b.device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(AtA_device[0, p], y_k[:, p])

            grad = (AtAy
                    - Atb
                    + y_k.sign() * tau_k
                    + U_k * sum_neighbors
                    + delta * rho_k)
            
            # Gradient clipping - only prevent extreme values
            max_grad_norm = 10.0  # Fixed, reasonable value
            grad = torch.clamp(grad, -max_grad_norm, max_grad_norm)
            
            # Check for NaN in gradient
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"Warning: NaN/Inf in gradient at iteration {k}, skipping update...")
                grad = torch.zeros_like(grad)
            
            # Update variables with stability checks
            y_next = y_k - alpha_k * grad
            
            # Value clipping - only prevent extreme values
            max_val = 100.0  # Fixed, reasonable value
            y_next = torch.clamp(y_next, -max_val, max_val)

            # Compute delta and update U_k
            delta = self.compute_delta(graph_list, y_next)
            delta = torch.clamp(delta, -20.0, 20.0)  # Reasonable delta bounds

            U_k = U_k + delta * eta_k
            U_k = torch.clamp(U_k, -100.0, 100.0)  # Fixed, reasonable bounds

            # Final NaN check before updating
            if torch.isnan(y_next).any() or torch.isinf(y_next).any():
                print(f"Warning: NaN/Inf in y_next at iteration {k}, using previous value...")
                y_next = y_k
            
            y_k = y_next
            Y.append(y_k)

        Y = torch.stack(Y)
        return Y, (alpha_k, tau_k, rho_k, eta_k)

    def compute_sum_neighbors(self, graph_list):
        batch_size = len(graph_list)
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        sum_neighbors = torch.zeros((batch_size, self.P, 1, 1), device=device)
        for i in range(batch_size):
            graph = graph_list[i]
            for p in range(self.P):
                sum_neighbors[i, p] = len(list(graph.neighbors(p)))
        return sum_neighbors

    def compute_Atx(self, x):
        device = x.device
        # Ensure A is on the same device as x
        A_device = self.A.to(device)
        Atx = torch.zeros((x.shape[0], self.P, self.n, x.shape[3]), device=device)
        for p in range(self.P):
            Atx[:, p] = torch.matmul(A_device[0, p].T, x[:, p])
        return Atx

    def compute_delta(self, graph_list, y1, y2=None):
        if y2 is None:
            y2 = y1
        delta = torch.zeros_like(y1)
        batch_size = len(graph_list)
        for b in range(batch_size):
            graph = graph_list[b]
            for p in range(self.P):
                y_p = y1[b, p]
                for j in graph.neighbors(p):
                    diff = y_p - y2[b, j]
                    delta[b, p] += diff
                    delta[b, j] -= diff
        return delta 
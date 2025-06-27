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


class DLASSO_GNNHyp3(nn.Module):
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
        
        # Better initialization for stability
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)  # Smaller gain for stability
        nn.init.zeros_(self.fc.bias)

        # Parameter bounds with better scaling
        self.alpha_max = torch.tensor(args.alpha_max)
        self.tau_max = torch.tensor(args.tau_max)
        self.rho_max = torch.tensor(args.rho_max)
        self.eta_max = torch.tensor(args.eta_max)
        
        # Global scale factor based on iteration count for stability
        # Higher iterations need smaller parameters to prevent instability
        base_iterations = 10.0  # Base reference point
        scale_factor = 1# min(1.0, base_iterations / self.K)
        self.alpha_max = self.alpha_max * scale_factor
        self.tau_max = self.tau_max * scale_factor
        self.rho_max = self.rho_max * scale_factor
        self.eta_max = self.eta_max * scale_factor

    def forward(self, b, graph_list):
        # Inputs
        # b: [batch_size,m*P]
        # graph_list: list of len batch_size, each item is graph with P agents
        batch_size = max(len(b), len(graph_list))
        K = self.K

        Atb = self.compute_Atx(b)
        sum_neighbors = self.compute_sum_neighbors(graph_list)

        Y = []
        # Initialize state variables
        y_k = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
        U_k = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
        delta = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)

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
            hyp = torch.clamp(hyp, min=1e-4, max=0.99)

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

            # Adaptive parameter clamping based on iteration number
            # Reduce parameter values for later iterations to prevent instability
            decay_factor = max(0.1, 1.0 - k / K)  # Decay from 1.0 to 0.1

            # alpha_k = torch.clamp(alpha_k * decay_factor, max=0.05)
            # tau_k = torch.clamp(tau_k * decay_factor, max=0.5)
            # rho_k = torch.clamp(rho_k * decay_factor, max=0.5)
            # eta_k = torch.clamp(eta_k * decay_factor, max=0.5)

            # Compute gradient
            # AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            # for p in range(self.P):
            #     AtAy[:, p] = torch.matmul(self.AtA[0, p], y_k[:, p])

            grad = (AtAy
                    - Atb
                    + y_k.sign() * tau_k
                    + U_k * sum_neighbors
                    + delta * rho_k)
            
            # Adaptive gradient clipping based on iteration - much more freedom
            max_grad_norm = max(30.0, 100.0 - k)  # Much higher clipping threshold over iterations
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

            delta = self.compute_delta(graph_list, y_next)
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
        return Y

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
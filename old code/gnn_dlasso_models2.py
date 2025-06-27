import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import math


class GNNHypernetwork2(nn.Module):
    def __init__(self, P, m, hidden_dim):
        super(GNNHypernetwork2, self).__init__()
        self.P = P
        self.m = m

        # GNN layers
        self.conv1 = GCNConv(self.m, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, 4 * hidden_dim)
        self.conv4 = GCNConv(4 * hidden_dim, 4 * hidden_dim)
        self.conv5 = GCNConv(4 * hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(4 * hidden_dim)
        
        # Add batch normalization for better training
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
        
        # Add residual connections and batch norm
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


class DLASSO_GNNHyp2(nn.Module):
    def __init__(self, A, args):
        super().__init__()
        # A [1, P, m, d]
        self.A = A
        _, self.P, self.m, self.n = self.A.shape

        self.AtA = self.compute_Atx(self.A)

        self.K = args.GHN_iter_num


        hidden_dim = args.GHyp_hidden
        self.DADMM_mode = args.DADMM_mode


        self.encoder = GNNHypernetwork2(P=self.P,
                                          m=self.n * 2,
                                          hidden_dim=hidden_dim)

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
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


        self.alpha_max = torch.tensor(args.alpha_max)
        self.tau_max = torch.tensor(args.tau_max)
        self.rho_max = torch.tensor(args.rho_max)
        self.eta_max = torch.tensor(args.eta_max)


    def forward(self, b, graph_list, K=None):
        # Inputs
        # b: [batch_size,m*P]
        # graph_list: list of len batch_size, each item is graph with P agents
        batch_size = max(len(b), len(graph_list))
        if K is None:
            K = self.K
        else:
            K = min(K, self.K)

        #b_enc = self.encoder_b(b, graph_list)
        Atb = self.compute_Atx(b)
        sum_neighbors = self.compute_sum_neighbors(graph_list)

        Y = []
        # Use zeros for initialization for stability - FIXED DEVICE ISSUE
        y_k = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
        U_k = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
        delta = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)

        for k in range(K):
            AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(self.AtA[0, p], y_k[:, p])
            hyp = torch.cat([AtAy,Atb],dim=2)
            hyp = self.encoder(hyp,graph_list)
            hyp = self.decoder(hyp)
            hyp = self.fc(hyp)
            hyp = torch.sigmoid(hyp)
            
            # Clamp hypernetwork output to prevent extreme values
            hyp = torch.clamp(hyp, min=1e-6, max=1.0)

            if self.DADMM_mode == 'same':
                hyp = hyp.view(batch_size,4, 1, 1, 1)
            else:
                hyp = hyp.view(batch_size,4, self.P, 1, 1)

            alpha_k = hyp[:,0]*self.alpha_max  # [b,P,1,1]
            tau_k = hyp[:,1]*self.tau_max  # [b,P,1,1]
            rho_k = hyp[:,2]*self.rho_max  # [b,P,1,1]
            eta_k = hyp[:,3]*self.eta_max  # [b,P,1,1]

            # Clamp parameters to reasonable ranges
            alpha_k = torch.clamp(alpha_k, max=0.1)
            tau_k = torch.clamp(tau_k, max=1.0)
            rho_k = torch.clamp(rho_k, max=1.0)
            eta_k = torch.clamp(eta_k, max=1.0)

            AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(self.AtA[0, p], y_k[:, p])

            grad = (AtAy
                    - Atb
                    + y_k.sign() * tau_k
                    + U_k * sum_neighbors
                    + delta * rho_k)
            
            # Clip gradients to prevent explosion
            grad = torch.clamp(grad, -10.0, 10.0)
            
            y_next = y_k - alpha_k * grad
            
            # Clip y_next to prevent extreme values
            y_next = torch.clamp(y_next, -100.0, 100.0)

            delta = self.compute_delta(graph_list, y_next)
            
            # Clip delta to prevent extreme values
            delta = torch.clamp(delta, -10.0, 10.0)

            U_k = U_k + delta * eta_k
            # Clip U_k to prevent extreme values
            U_k = torch.clamp(U_k, -100.0, 100.0)
            
            y_k = y_next
            Y.append(y_k)

        Y = torch.stack(Y)
        return Y



    def compute_sum_neighbors(self, graph_list):
        batch_size = len(graph_list)
        # Get device from the first graph (assuming all are on same device)
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        sum_neighbors = torch.zeros((batch_size, self.P, 1, 1), device=device)  # [batch_size,P,1,1]
        for i in range(batch_size):
            graph = graph_list[i]
            for p in range(self.P):
                sum_neighbors[i, p] = len(list(graph.neighbors(p)))
        return sum_neighbors

    def compute_Atx(self, x):
        device = x.device
        Atx = torch.zeros((x.shape[0], self.P, self.n, x.shape[3]), device=device)
        for p in range(self.P):
            Atx[:, p] = torch.matmul(self.A[0, p].T, x[:, p])
        return Atx

    def compute_delta(self, graph_list, y1, y2=None):
        if y2 is None:
            y2 = y1
        delta = torch.zeros_like(y1)  # [b,P,d,1]
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

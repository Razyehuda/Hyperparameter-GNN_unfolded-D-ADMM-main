import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import math


class GNNHypernetwork(nn.Module):
    def __init__(self, P, m, hidden_dim, hyp_shape, upper_limit=3.0):
        super(GNNHypernetwork, self).__init__()
        self.P = P
        self.m = m
        self.hyp_shape = hyp_shape  # [K,P,4]
        self.upper_limit = upper_limit

        self.encoder = nn.Sequential(
            nn.Linear(self.P * self.m, P * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(P * hidden_dim, 2 * P * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * P * hidden_dim, 4 * P * hidden_dim)
        )

        # GNN layers
        self.conv1 = GCNConv(4 * hidden_dim, 2 * hidden_dim)
        self.conv2 = GCNConv(2 * hidden_dim, 2 * hidden_dim)

        # Final MLP to produce parameters
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, math.prod(hyp_shape))

        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-7)
        nn.init.constant_(self.fc2.bias, -10.0)

    def forward(self, b, graph_list):
        # b: [batch_size, P,m]
        batch_size = b.shape[0]
        x = b.view(batch_size, -1)
        x = self.encoder(x)
        x = x.view(batch_size, self.P, -1)
        hyp = torch.stack([self.get_hyp(x[i], graph_list[i]) for i in range(batch_size)])
        hyp = torch.cumsum(hyp, dim=1)
        return torch.sigmoid(hyp)

    def get_hyp(self, node_features, nx_graph):
        edge_index = from_networkx(nx_graph).edge_index

        x = F.leaky_relu(self.conv1(node_features, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))

        # Global mean pooling (since there's only one graph, no batch vector needed)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # all nodes belong to graph 0
        x = global_mean_pool(x, batch)  # [1, hidden_dim]

        # Output positive weights using softplus
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x).squeeze(0)  # [K * P * A]
        x = x.view(self.hyp_shape)  #+self.hyp_bias
        return x


class DLASSO_GNNHyp(nn.Module):
    def __init__(self, A, args):
        super().__init__()
        # A [1, P, m, d]
        self.A = A
        _, self.P, self.m, self.n = self.A.shape

        self.AtA = self.compute_Atx(self.A)

        self.K = args.GHN_iter_num

        self.DADMM_mode = args.DADMM_mode
        if args.DADMM_mode == 'same':
            hyp_shape = [self.K, 1, 4]
        else:
            hyp_shape = [self.K, self.P, 4]

        self.gnn_hyp = GNNHypernetwork(P=self.P,
                                       m=self.m,
                                       hidden_dim=args.GHyp_hidden,
                                       hyp_shape=hyp_shape)
        self.max_param = torch.tensor([[[args.alpha_max, args.tau_max, args.rho_max, args.eta_max]]]).unsqueeze(0)


    def forward(self, b, graph_list, K=None):
        # Inputs
        # b: [batch_size,m*P]
        # graph_list: list of len batch_size, each item is graph with P agents
        batch_size = max(len(b), len(graph_list))
        if K is None:
            K = self.K
        else:
            K = min(K, self.K)

        hyp = (self.gnn_hyp(b, graph_list)*self.max_param).unsqueeze(-1).unsqueeze(-1)
        Atb = self.compute_Atx(b)
        sum_neighbors = self.compute_sum_neighbors(graph_list)

        Y = []
        y_k = torch.randn((batch_size, self.P, self.n, 1)) * 1e-2  # [batch_size,P,d,1]
        U_k = torch.randn((batch_size, self.P, self.n, 1)) * 1e-2  # [batch_size,P,d,1]
        delta = torch.randn((batch_size, self.P, self.n, 1)) * 1e-2

        for k in range(K):
            alpha_k = hyp[:, k, :, 0]  # [b,P,1,1]
            tau_k = hyp[:, k, :, 1]  # [b,P,1,1]
            rho_k = hyp[:, k, :, 2]  # [b,P,1,1]
            eta_k = hyp[:, k, :, 3]  # [b,P,1,1]

            AtAy = torch.zeros((batch_size, self.P, self.n, 1))
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(self.AtA[0, p], y_k[:, p])

            grad = (AtAy
                    - Atb
                    + y_k.sign() * tau_k
                    + U_k * sum_neighbors
                    + delta * rho_k)
            y_next = y_k - alpha_k * grad

            delta = self.compute_delta(graph_list, y_next)

            U_k = U_k + delta * eta_k
            y_k = y_next
            Y.append(y_k)

        Y = torch.stack(Y)
        return Y

    def compute_sum_neighbors(self, graph_list):
        batch_size = len(graph_list)
        sum_neighbors = torch.zeros((batch_size, self.P, 1, 1))  # [batch_size,P,1,1]
        for i in range(batch_size):
            graph = graph_list[i]
            for p in range(self.P):
                sum_neighbors[i, p] = len(list(graph.neighbors(p)))
        return sum_neighbors

    def compute_Atx(self, x):
        Atx = torch.zeros((x.shape[0], self.P, self.n, x.shape[3]))
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

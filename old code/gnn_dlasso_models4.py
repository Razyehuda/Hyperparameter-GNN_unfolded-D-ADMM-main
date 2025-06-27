import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import from_networkx
import math


class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network Layer
    """
    def __init__(self, in_channels, out_channels, edge_channels=None):
        super(MPNNLayer, self).__init__(aggr='add')  # Use sum aggregation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        
        # Message function (edge features to messages)
        if edge_channels is not None:
            self.message_net = nn.Sequential(
                nn.Linear(2 * in_channels + edge_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_channels, out_channels)
            )
        else:
            self.message_net = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_channels, out_channels)
            )
        
        # Update function (node features + aggregated messages)
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.message_net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        for layer in self.update_net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr=None):
        # x_i: source node features
        # x_j: target node features
        if edge_attr is not None:
            message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            message_input = torch.cat([x_i, x_j], dim=-1)
        
        return self.message_net(message_input)
    
    def update(self, aggr_out, x):
        # aggr_out: aggregated messages
        # x: original node features
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class GNNHypernetwork4(nn.Module):
    def __init__(self, P, m, hidden_dim):
        super(GNNHypernetwork4, self).__init__()
        self.P = P
        self.m = m

        # MPNN layers with increasing complexity
        self.mpnn1 = MPNNLayer(self.m, hidden_dim)
        self.mpnn2 = MPNNLayer(hidden_dim, 2 * hidden_dim)
        self.mpnn3 = MPNNLayer(2 * hidden_dim, 4 * hidden_dim)
        self.mpnn4 = MPNNLayer(4 * hidden_dim, 4 * hidden_dim)
        self.mpnn5 = MPNNLayer(4 * hidden_dim, 4 * hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)
        self.bn4 = nn.BatchNorm1d(4 * hidden_dim)
        self.bn5 = nn.BatchNorm1d(4 * hidden_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(4 * hidden_dim)
        
        # Residual connections
        self.residual1 = nn.Linear(self.m, hidden_dim) if self.m != hidden_dim else nn.Identity()
        self.residual2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.residual3 = nn.Linear(2 * hidden_dim, 4 * hidden_dim)
        self.residual4 = nn.Linear(4 * hidden_dim, 4 * hidden_dim)
        self.residual5 = nn.Linear(4 * hidden_dim, 4 * hidden_dim)

    def forward(self, x, graph_list):
        batch_size = x.shape[0]
        hyp = torch.stack([self.graph_conv(x[i], graph_list[i]) for i in range(batch_size)])
        return hyp.view(batch_size, -1)

    def graph_conv(self, x, nx_graph):
        edge_index = from_networkx(nx_graph).edge_index
        x = x.squeeze(-1)
        
        # MPNN forward pass with residual connections
        # Layer 1
        x1 = self.mpnn1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1 + self.residual1(x))  # Residual connection
        x1 = self.dropout(x1)
        
        # Layer 2
        x2 = self.mpnn2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.residual2(x1))  # Residual connection
        x2 = self.dropout(x2)
        
        # Layer 3
        x3 = self.mpnn3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + self.residual3(x2))  # Residual connection
        x3 = self.dropout(x3)
        
        # Layer 4
        x4 = self.mpnn4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4 + self.residual4(x3))  # Residual connection
        x4 = self.dropout(x4)
        
        # Layer 5
        x5 = self.mpnn5(x4, edge_index)
        x5 = self.bn5(x5)
        x5 = F.relu(x5 + self.residual5(x4))  # Residual connection
        x5 = self.norm(x5)
        
        return x5.view(-1)


class AttentionMPNNLayer(MessagePassing):
    """
    Attention-based Message Passing Neural Network Layer
    """
    def __init__(self, in_channels, out_channels, heads=4):
        super(AttentionMPNNLayer, self).__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads
        
        # Multi-head attention
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        
        for layer in self.message_net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        for layer in self.update_net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # Compute attention weights
        q = self.query(x_i).view(-1, self.heads, self.head_dim)
        k = self.key(x_j).view(-1, self.heads, self.head_dim)
        v = self.value(x_j).view(-1, self.heads, self.head_dim)
        
        # Scaled dot-product attention
        attention_weights = torch.sum(q * k, dim=-1) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_weights, dim=-1).unsqueeze(-1)
        
        # Apply attention to values
        attended_values = (v * attention_weights).view(-1, self.out_channels)
        
        # Combine with message function
        message_input = torch.cat([x_i, x_j], dim=-1)
        messages = self.message_net(message_input)
        
        return messages + attended_values
    
    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class DLASSO_GNNHyp4(nn.Module):
    def __init__(self, A, args):
        super().__init__()
        # A [1, P, m, d]
        self.A = A
        _, self.P, self.m, self.n = self.A.shape

        self.AtA = self.compute_Atx(self.A)
        self.K = args.GHN_iter_num
        hidden_dim = args.GHyp_hidden
        self.DADMM_mode = args.DADMM_mode

        # MPNN-based encoder
        self.encoder = GNNHypernetwork4(P=self.P,
                                       m=self.n * 2,
                                       hidden_dim=hidden_dim)

        # Improved decoder with attention
        self.decoder = nn.Sequential(
            nn.Linear(self.P * 4 * hidden_dim, 4 * hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for parameter generation
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1)
        
        if args.DADMM_mode == 'same':
            self.fc = nn.Linear(hidden_dim, 4)
        else:
            self.fc = nn.Linear(hidden_dim, 4 * self.P)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # Parameter bounds
        self.alpha_max = torch.tensor(args.alpha_max)
        self.tau_max = torch.tensor(args.tau_max)
        self.rho_max = torch.tensor(args.rho_max)
        self.eta_max = torch.tensor(args.eta_max)

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
            AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(self.AtA[0, p], y_k[:, p])
            
            # Generate hyperparameters with MPNN
            hyp = torch.cat([AtAy, Atb], dim=2)
            hyp = self.encoder(hyp, graph_list)
            
            # Apply attention mechanism
            hyp_reshaped = hyp.view(batch_size, self.P, -1)  # [batch_size, P, hidden_dim]
            hyp_reshaped = hyp_reshaped.transpose(0, 1)  # [P, batch_size, hidden_dim]
            hyp_attended, _ = self.attention(hyp_reshaped, hyp_reshaped, hyp_reshaped)
            hyp_attended = hyp_attended.transpose(0, 1)  # [batch_size, P, hidden_dim]
            hyp = hyp_attended.view(batch_size, -1)  # Flatten back
            
            hyp = self.decoder(hyp)
            hyp = self.fc(hyp)
            hyp = torch.sigmoid(hyp)
            
            # Clamp hypernetwork output
            hyp = torch.clamp(hyp, min=1e-6, max=1.0)

            if self.DADMM_mode == 'same':
                hyp = hyp.view(batch_size, 4, 1, 1, 1)
            else:
                hyp = hyp.view(batch_size, 4, self.P, 1, 1)

            # Extract parameters
            alpha_k = hyp[:, 0] * self.alpha_max
            tau_k = hyp[:, 1] * self.tau_max
            rho_k = hyp[:, 2] * self.rho_max
            eta_k = hyp[:, 3] * self.eta_max

            # Clamp parameters
            alpha_k = torch.clamp(alpha_k, max=0.1)
            tau_k = torch.clamp(tau_k, max=1.0)
            rho_k = torch.clamp(rho_k, max=1.0)
            eta_k = torch.clamp(eta_k, max=1.0)

            # Compute gradient
            AtAy = torch.zeros((batch_size, self.P, self.n, 1), device=b.device)
            for p in range(self.P):
                AtAy[:, p] = torch.matmul(self.AtA[0, p], y_k[:, p])

            grad = (AtAy
                    - Atb
                    + y_k.sign() * tau_k
                    + U_k * sum_neighbors
                    + delta * rho_k)
            
            # Clip gradients
            grad = torch.clamp(grad, -10.0, 10.0)
            
            # Update variables
            y_next = y_k - alpha_k * grad
            y_next = torch.clamp(y_next, -100.0, 100.0)

            delta = self.compute_delta(graph_list, y_next)
            delta = torch.clamp(delta, -10.0, 10.0)

            U_k = U_k + delta * eta_k
            U_k = torch.clamp(U_k, -100.0, 100.0)
            
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
        Atx = torch.zeros((x.shape[0], self.P, self.n, x.shape[3]), device=device)
        for p in range(self.P):
            Atx[:, p] = torch.matmul(self.A[0, p].T, x[:, p])
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
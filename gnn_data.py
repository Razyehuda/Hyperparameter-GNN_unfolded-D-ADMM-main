from torch.utils.data import Dataset
import torch
import numpy as np


def set_Data(A, data_len, args):
    device = A.device  # Get the device of A tensor
    sigma = torch.pow(10, torch.tensor(-args.snr / 40, device=device))
    _, P, m, n = A.shape

    y = 2 * torch.randn(data_len, n, 1, device=device) * (torch.rand(data_len, n, 1, device=device) <= 0.25)
    b = torch.randn(data_len, P, m, 1, device=device) * sigma
    for p in range(P):
        b[:, p] = torch.matmul(A[0, p], y)
    return torch.utils.data.DataLoader(GNN_Data(b, y), batch_size=args.batch_size, shuffle=True, drop_last=True)


class GNN_Data(Dataset):
    def __init__(self, b, y):
        self.b = b  #input [data_len,m]
        self.y = y  #input [data_len,n]

    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, item):
        return self.b[item], self.y[item]

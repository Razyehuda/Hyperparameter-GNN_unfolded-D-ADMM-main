from torch.utils.data import Dataset
import torch
import numpy as np
import os


class SimulatedData(Dataset):
    def __init__(self, idx, snr):
        # Construct the relative path to the data file
        data_path = os.path.join('data', f'data_{snr}_snr.npy')
        
        with open(data_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            label = np.load(f, allow_pickle=True)
        
        data = data[:1200]
        label = label[:1200]
        
        if idx >= 0.7 * data.shape[0]:
            self.x = torch.from_numpy(data[:idx, :, :]).to(torch.float32)
            self.y = torch.from_numpy(label[:idx, :, :]).to(torch.float32)
        else:
            self.x = torch.from_numpy(data[-idx:, :, :]).to(torch.float32)
            self.y = torch.from_numpy(label[-idx:, :, :]).to(torch.float32)
        self.samples_num = idx

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples_num
import torch
import torch.nn as nn

class TieredMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('hot_data', torch.zeros(1000, 256))  # SSD
        self.register_buffer('cold_data', torch.zeros(10000, 256))  # HDD
        self.access_counter = torch.zeros(10000)

    def promote(self, indices):
        # Move frequent data to SSD
        mask = self.access_counter[indices] > 5
        self.hot_data = torch.cat([self.hot_data, self.cold_data[indices][mask]])
        self.cold_data[indices[mask]] = 0

    def demote(self):
        # Move stale data to HDD
        stale = self.access_counter[self.hot_indices] < 2
        self.cold_data = torch.cat([self.cold_data, self.hot_data[stale]])
        self.hot_data = self.hot_data[~stale]

import torch
import torch.nn as nn

class MemoryBank(nn.Module):
    def __init__(self, encoding_dim, capacity=1000):
        super().__init__()
        self.capacity = capacity
        self.memory = nn.Parameter(torch.zeros(capacity, encoding_dim), requires_grad=False).cuda()
        self.usage_count = nn.Parameter(torch.zeros(capacity), requires_grad=False).cuda()
        self.importance_score = nn.Parameter(torch.zeros(capacity), requires_grad=False).cuda()
        self.temporal_index = nn.Parameter(torch.zeros(capacity), requires_grad=False).cuda()
        self.next_index = 0

    def add(self, encoded_experience):
        if self.next_index < self.capacity:
            self._add_new_memory(encoded_experience)
        else:
            self._replace_memory(encoded_experience)

    def _add_new_memory(self, encoded_experience):
        self.memory.data[self.next_index] = encoded_experience.data
        self.usage_count.data[self.next_index] = 1
        self.importance_score.data[self.next_index] = self._calculate_importance(encoded_experience)
        self.temporal_index.data[self.next_index] = self.next_index
        self.next_index += 1

    def _replace_memory(self, encoded_experience):
        # Find least important memory
        combined_score = self.importance_score * self.usage_count
        replace_idx = torch.argmin(combined_score[:self.capacity])
        
        self.memory.data[replace_idx] = encoded_experience.data
        self.usage_count.data[replace_idx] = 1
        self.importance_score.data[replace_idx] = self._calculate_importance(encoded_experience)
        self.temporal_index.data[replace_idx] = self.next_index
        self.next_index += 1

    def _calculate_importance(self, experience):
        # Simple importance calculation based on vector magnitude
        return torch.norm(experience.data)

    def get_memory(self):
        return self.memory[:self.next_index]

    def get_usage_counts(self):
        return self.usage_count[:self.next_index]
    
    def get_importance_scores(self):
        return self.importance_score[:self.next_index]

    def __len__(self):
        return self.next_index
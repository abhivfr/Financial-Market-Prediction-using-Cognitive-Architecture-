import torch
import torch.nn as nn
import torch.nn.functional as F

class ConflictEvaluator(nn.Module):
    def __init__(self, input_dim=4, memory_dim=256, threshold=0.3):
        super().__init__()
        self.threshold = threshold
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.input_projection = nn.Linear(input_dim, memory_dim)
        
        # Add conflict tracking
        self.register_buffer('conflict_history', torch.zeros(100))
        self.conflict_ptr = 0
        
        # Add exploration parameter
        self.exploration_weight = nn.Parameter(torch.tensor(0.1))

    def evaluate(self, current, retrieved):
        projected_current = self.input_projection(current)
        projected_current_normalized = F.normalize(projected_current, dim=-1)
        retrieved_normalized = F.normalize(retrieved, dim=-1)
        
        similarity = torch.sum(projected_current_normalized * retrieved_normalized, dim=-1)
        conflict = similarity < self.threshold
        
        # Update conflict history
        if self.training:
            conflict_rate = conflict.float().mean().item()
            self.conflict_history[self.conflict_ptr % 100] = conflict_rate
            self.conflict_ptr += 1
            
            # Adjust exploration weight based on recent conflict rate
            recent_conflict_rate = self.conflict_history.mean().item()
            # Higher conflicts → increase exploration
            if recent_conflict_rate > 0.5:
                self.exploration_weight.data *= 1.01  # Gradually increase exploration
            else:
                self.exploration_weight.data *= 0.99  # Gradually decrease exploration
            
            # Keep exploration weight in reasonable bounds
            self.exploration_weight.data.clamp_(0.01, 0.5)
            
        return conflict, similarity, self.exploration_weight.item()

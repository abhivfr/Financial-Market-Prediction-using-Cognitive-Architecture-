import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F # Import F for normalization

class NeuralDictionary(nn.Module):
    # Added financial_retention parameter with a default value
    # Adjusted default key_dim and value_dim to align with CognitiveArchitecture default hidden_dim=256 if not specified
    def __init__(self, key_dim=256, value_dim=256, capacity=1000, financial_retention=0.9):
        super().__init__()
        self.capacity = capacity
        self.retention = financial_retention # Use the passed parameter

        # Initialize all tensors as nn.Parameters to ensure they're on the correct device
        self.keys = nn.Parameter(torch.randn(capacity, key_dim))
        self.values = nn.Parameter(torch.randn(capacity, value_dim))
        self.usage = nn.Parameter(torch.zeros(capacity))
        self.market_importance = nn.Parameter(torch.ones(4))
        self.importance_scores = nn.Parameter(torch.ones(capacity))
        
        # Register buffers for non-parameter tensors that should still move to GPU
        # Specify dtype explicitly for counters
        self.register_buffer('access_counts', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('last_access', torch.zeros(capacity, dtype=torch.long))

        # Initialize next_index to track filled memory slots
        self.next_index = 0

        # 4D specific parameters - market_importance should remain 4D
        self.temporal_decay = 0.95
        self.last_update = None

        # Initialize event detection support
        self.recent_keys = []
        self.average_std = 1.0  # Initial guess, will adapt
        self.last_consolidation = 0

        # Add market regime tracking
        self.regime_history = []
        self.regime_window = 50

        # Temperature parameter for similarity scaling
        self.similarity_temp = nn.Parameter(torch.ones(1) * 1.5)
        # Add regime embeddings
        self.regime_embeddings = nn.Parameter(torch.randn(3, 256))  # 3 regimes, 256D embedding

        # Add these lines for caching
        self.query_cache = []
        self.max_cache_size = 10

    def update_usage(self, indices):
        # Ensure indices is on the correct device and dtype
        indices = indices.to(self.usage.device)
        
        # Update usage (float tensor)
        self.usage.data[indices] += 1.0
        
        # Update access counts (long tensor)
        self.access_counts[indices] += 1
        
        # Update last access with current index (long tensor)
        current_index = torch.tensor(self.next_index, 
                                   dtype=torch.long,
                                   device=indices.device)
        self.last_access[indices] = current_index

    def add(self, key, value):
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        batch_size = key.size(0)
        # Add safety check
        if batch_size > self.capacity:
            print(f"Warning: Batch size {batch_size} exceeds memory capacity {self.capacity}. Truncating batch.")
            key = key[:self.capacity]
            value = value[:self.capacity]
            batch_size = self.capacity

        # Ensure key and values are on the same device as self.keys and self.values
        key = key.to(self.keys.device)
        value = value.to(self.values.device)

        if self.next_index + batch_size <= self.capacity:
            # Memory has enough space, add to the next available slots
            indices = torch.arange(self.next_index, 
                                 self.next_index + batch_size,
                                 dtype=torch.long,
                                 device=self.keys.device)
            self.next_index += batch_size # Update next_index after adding

        else:
            # Memory does not have enough space for the full batch
            # Implement a replacement strategy: replace the 'batch_size' least used items.
            # Find indices of the least used slots to replace
            _, indices = torch.topk(self.usage.data, 
                                  k=batch_size,
                                  largest=False)

        # Add new memories at the determined indices
        self.keys.data[indices] = key.detach()
        self.values.data[indices] = value.detach()
        self.usage.data[indices] = 1.0 # Reset usage for the replaced/added items to high

        self.update_usage(indices) # Update usage (increment + decay, though decay is currently commented out)

        # Track recent keys for event detection
        if key.dim() == 1:
            self.recent_keys.append(key.detach().clone())
        else:
            # Take the mean if it's a batch
            self.recent_keys.append(key.mean(dim=0).detach().clone())
        
        # Keep recent_keys to a reasonable size
        if len(self.recent_keys) > 100:
            self.recent_keys.pop(0)
        
        # Check if we should consolidate based on events
        # Only check periodically to avoid overhead
        if self.next_index % 100 == 0 and len(self.recent_keys) > 10:
            self.consolidate_financial_memory(force=False)

        return indices

    def retrieve(self, query, top_k=3, regime=None):
        """Memory-optimized retrieval with caching"""
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Ensure query and keys are on the same device and are float type
        query = query.to(self.keys.device).float()
        
        # Check cache for identical queries
        if hasattr(self, 'query_cache'):
            cache_hit = False
            for cached_query, cached_result, cached_regime in self.query_cache:
                if torch.allclose(query, cached_query, rtol=1e-4, atol=1e-4):
                    # If regime is part of the cache key, check it too
                    if regime is None or cached_regime is None or torch.allclose(regime, cached_regime, rtol=1e-3, atol=1e-3):
                        cache_hit = True
                        retrieved_values, attention_var = cached_result
                        
                        # Update usage counts but return cached result
                        if hasattr(self, 'similarity_scores'):
                            _, indices = torch.topk(self.similarity_scores, k=min(top_k, self.next_index), dim=-1)
                            self.update_usage(indices.flatten())
                            self.update_importance(indices.flatten())
                        
                        return retrieved_values, attention_var
        
        # Only consider the part of keys that has been filled
        keys = self.keys[:self.next_index].to(self.keys.device).float()
        values = self.values[:self.next_index].to(self.values.device).float()

        # Check if keys is empty based on its first dimension
        if keys.size(0) == 0:
            # Return empty tensors if memory is empty
            return torch.empty(query.size(0), 0, self.values.size(1), device=query.device), torch.tensor(0.0, device=query.device)

        # Incorporate regime information if provided
        if regime is not None and self.regime_embeddings is not None:
            regime_weights = F.softmax(regime, dim=-1)
            regime_encoded = torch.matmul(regime_weights, self.regime_embeddings)
            # Blend query with regime context
            query = query + 0.2 * regime_encoded

        # Normalize with numerical stability
        query_norm = F.normalize(query, p=2, dim=-1, eps=1e-8)
        keys_norm = F.normalize(keys, p=2, dim=-1, eps=1e-8)
        
        # Compute similarities with temperature scaling
        similarity_scores = torch.matmul(query_norm, keys_norm.T)
        similarity_scores = similarity_scores / self.similarity_temp
        
        # Compute attention variance across similarity scores
        attention_var = similarity_scores.var(dim=-1).mean()

        # Find the top-k memories with adaptive k
        k_actual = min(top_k, keys.size(0))
        _, indices = torch.topk(similarity_scores, k=k_actual, dim=-1)

        # Retrieve values with importance weighting
        retrieved_values = values[indices]
        importance_weights = self.importance_scores[indices].unsqueeze(-1)
        weighted_values = retrieved_values * importance_weights

        # Update usage stats
        self.update_usage(indices.flatten())
        self.update_importance(indices.flatten())

        # Before returning, add to cache for future lookups
        if hasattr(self, 'query_cache'):
            if len(self.query_cache) >= self.max_cache_size:
                self.query_cache.pop(0)  # Remove oldest entry
            self.query_cache.append((query.detach().clone(), 
                                 (retrieved_values.detach().clone(), attention_var.detach().clone()),
                                 regime.detach().clone() if regime is not None else None))
        
        return retrieved_values, attention_var

    def consolidate_financial_memory(self, force=False):
        """Improved memory consolidation with importance and recency weighting"""
        # Identify important memories
        if self.next_index == 0:
            # No memories to consolidate
            return False
        
        # Add event detection
        event_triggered = False
        
        if hasattr(self, 'recent_keys') and len(self.recent_keys) > 10:
            try:
                recent_keys = torch.stack(self.recent_keys[-10:])
                
                # Calculate volatility in recent inputs
                mean_recent = recent_keys.mean(dim=0)
                std_recent = recent_keys.std(dim=0)
                
                # Detect market volatility spike - potential regime change
                if std_recent.mean() > 1.5 * self.average_std:
                    event_triggered = True
                    print("🔍 Event detected: Market volatility spike - consolidating memory")
                
                # Update running average of standard deviation
                self.average_std = 0.9 * self.average_std + 0.1 * std_recent.mean()
            except Exception as e:
                print(f"Warning: Error in event detection: {e}")
                return False

        # Only consolidate if forced or event triggered
        if not (force or event_triggered):
            return False
        
        # Enhanced importance calculation
        # Combine importance score, access frequency, and recency
        importance = self.importance_scores[:self.next_index]
        access_freq = torch.log1p(self.access_counts[:self.next_index].float())
        
        # Calculate recency score (higher for more recent)
        max_index = float(self.next_index) if self.next_index > 0 else 1.0
        recency = 1.0 - (self.next_index - self.last_access[:self.next_index].float()) / max_index
        
        # Combine scores with weights for importance (0.5), access (0.3), recency (0.2)
        combined_score = 0.5 * importance + 0.3 * access_freq + 0.2 * recency
        
        # Keep most valuable memories
        num_to_retain = int(len(combined_score) * self.retention)
        if num_to_retain > 0:
            _, indices = torch.topk(combined_score, k=num_to_retain)
            
            # Compress and reorganize memory
            self.keys.data[:num_to_retain] = self.keys[indices]
            self.values.data[:num_to_retain] = self.values[indices]
            self.importance_scores.data[:num_to_retain] = self.importance_scores[indices]
            self.access_counts[:num_to_retain] = self.access_counts[indices]
            self.last_access[:num_to_retain] = self.last_access[indices]
            
            # Reset the rest
            if num_to_retain < self.capacity:
                self.keys.data[num_to_retain:] = 0
                self.values.data[num_to_retain:] = 0
                self.importance_scores.data[num_to_retain:] = 1
                self.access_counts[num_to_retain:] = 0
                self.last_access[num_to_retain:] = 0
            
            self.next_index = num_to_retain
        
        return True

    def get_memory_stats(self):
        # Ensure usage, keys, and market_importance are on the CPU for numpy conversion if needed, or handle on GPU
        # Calculate stats based on the filled portion of memory
        current_keys = self.keys[:self.next_index]
        current_usage = self.usage[:self.next_index]
        current_values = self.values[:self.next_index]


        return {
            'usage_mean': current_usage.mean().item() if len(current_usage) > 0 else 0.0,
            'usage_std': current_usage.std().item() if len(current_usage) > 0 else 0.0,
            'key_norm_mean': torch.norm(current_keys, dim=1).mean().item() if len(current_keys) > 0 else 0.0,
            'value_norm_mean': torch.norm(current_values, dim=1).mean().item() if len(current_values) > 0 else 0.0,
            'market_importance': self.market_importance.tolist(), # This is still relevant to show the 4D weights
            'filled_capacity': self.next_index
        }

    def __len__(self):
        return max(0, self.next_index)  # Ensure non-negative

    def update_importance(self, indices, rewards=None):
        # Ensure indices is on the correct device
        indices = indices.to(self.importance_scores.device)
        
        if rewards is None:
            # Default importance update based on access
            self.importance_scores.data[indices] *= 1.1
        else:
            # Ensure rewards is on the correct device
            rewards = rewards.to(self.importance_scores.device)
            self.importance_scores.data[indices] *= (1.0 + rewards)

import torch
import torch.nn.functional as F
import torch.nn as nn  # Ensure nn is imported

class MemoryRetrieval(nn.Module):  # Inherit from nn.Module
    def __init__(self):
        super().__init__()

    def retrieve(self, query, memory_bank):
        # Unpack keys and values from memory bank
        keys, values = memory_bank.get_memory()

        # Check if keys is empty based on its first dimension
        if keys.size(0) == 0:
            return None, None

        query = query.float()
        keys = keys.float()
        values = values.float()

        # Calculate cosine similarity between the query and keys
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        similarity_scores = torch.matmul(query_norm, keys_norm.T)

        # Compute attention variance across similarity scores
        attention_var = similarity_scores.var(dim=-1).mean()

        # Find the index of the best match
        best_match_index = torch.argmax(similarity_scores, dim=-1)

        # Retrieve the corresponding value from values tensor
        retrieved = values[best_match_index.squeeze(0)]

        return retrieved, attention_var

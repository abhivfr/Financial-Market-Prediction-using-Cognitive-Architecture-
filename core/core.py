import torch
from torch.nn import Module, Parameter, Linear
from src.utils.memory_guard import MemoryGuard
from src.memory.buffer import MemoryBuffer
from src.memory.encoding import MemoryEncoder
from src.core.memory import NeuralDictionary
from src.memory.retrieval import MemoryRetrieval
from src.memory.replay import MemoryReplay
from src.memory.conflict import ConflictEvaluator
from src.memory.confidence import MemoryConfidence
from src.monitoring.core_monitor import CoreMonitor

class ConsciousnessCore(Module):
    def __init__(self, dim=128, max_depth=3, memory_buffer_capacity=100, 
                 encoding_dim=128, adaptive_depth=True, financial_dim=4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_counter = 0
        self.max_depth = max_depth
        self.adaptive_depth = adaptive_depth
        self.register_buffer('grad_norms', torch.zeros(max_depth))
        self.memory_guard = MemoryGuard()
        self.monitor = CoreMonitor()

        # Financial feature integration
        self.financial_proj = Linear(financial_dim, dim).to(self.device)
        
        # Core parameters
        self.theta = Parameter(torch.randn(dim, dim, device=self.device) * 0.02)
        self.gamma = Parameter(torch.ones(1, device=self.device) * 0.9)

        # Memory system components
        self.memory_buffer = MemoryBuffer(capacity=memory_buffer_capacity)
        self.memory_encoder = MemoryEncoder(input_dim=dim, encoding_dim=encoding_dim).to(self.device)
        self.memory_bank = NeuralDictionary(dim=encoding_dim).to(self.device)
        self.memory_retrieval = MemoryRetrieval().to(self.device)
        self.memory_replay = MemoryReplay().to(self.device)
        self.memory_projection = Linear(encoding_dim, dim).to(self.device)

        # Conflict resolution
        self.conflict_evaluator = ConflictEvaluator(threshold=0.3)
        self.memory_confidence = MemoryConfidence()

        # Monitoring
        self.attention_var = None

    def forward(self, x, financial_feat=None, depth=0):
        self.monitor.start_forward()
        x = x.to(self.device)

        # Financial feature fusion
        if financial_feat is not None:
            projected_finance = self.financial_proj(financial_feat.to(self.device))
            x = x + projected_finance

        current_max_depth = self._calculate_max_depth(x)
        if depth >= current_max_depth:
            self.monitor.end_forward()
            return x

        self.depth_counter = max(self.depth_counter, depth + 1)
        x = self.memory_guard.check(x).to(self.device)
        x.retain_grad()

        # Memory processing pipeline
        encoded_experience = self.memory_encoder(x.detach())
        self.memory_buffer.add(encoded_experience)
        self.memory_bank.add(encoded_experience)

        # Consolidated memory retrieval
        if len(self.memory_buffer) % 100 == 0:
            self.memory_bank.consolidate()

        retrieved_memory, attention_var = self.memory_retrieval.retrieve(
            encoded_experience.unsqueeze(0), self.memory_bank
        )
        self.attention_var = attention_var
        self.monitor.log_attention(attention_var)

        # Conflict resolution
        conflict_mask, _ = self.conflict_evaluator.evaluate(encoded_experience, retrieved_memory)
        retrieved_memory[conflict_mask] = 0
        replayed_memory = self.memory_replay.replay(retrieved_memory)

        # Core transformation
        x_transformed = torch.matmul(x, self.theta)
        x_transformed = torch.nn.functional.gelu(x_transformed)

        # Memory integration
        if replayed_memory is not None:
            projected_encoded = self.memory_projection(encoded_experience)
            x_transformed += projected_encoded

        # Resource-aware recursion control
        signal = x_transformed.norm().item()
        vram_usage = torch.cuda.memory_allocated() / 1e9

        if signal < 0.1 or vram_usage > torch.cuda.get_device_properties(0).total_memory * 0.8 / 1e9:
            self.monitor.end_forward(activations=x_transformed)
            return x_transformed

        self.monitor.end_forward(activations=x_transformed)
        return self.forward(x_transformed, financial_feat, depth + 1)

    # Rest of the class remains unchanged
    def _calculate_max_depth(self, x):
        if self.adaptive_depth:
            base_signal = x.norm().item()
            dynamic_depth = min(self.max_depth, int(2 + (base_signal * 3)))
            return max(1, dynamic_depth)
        return self.max_depth

    def backward(self, loss):
        self.monitor.start_backward()
        loss.backward()
        self.clip_gradients()
        self.monitor.end_backward(gradients=self.theta.grad)

    def clip_gradients(self, max_norm=1.0):
        grad_norm = torch.norm(self.theta.grad)
        final_norm = max_norm if (grad_norm > max_norm or torch.isnan(grad_norm)) else grad_norm.item()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.grad_norms[self.depth_counter - 1] = final_norm

    def get_monitoring_stats(self):
        return {
            'core_metrics': self.monitor.get_statistics(),
            'memory_status': self.memory_guard.get_memory_status(),
            'depth': self.depth_counter,
            'attention_variance': self.attention_var.item() if self.attention_var else None
        }

    def reset_monitoring(self):
        self.monitor.reset()
        self.depth_counter = 0
        self.attention_var = None
class CognitiveConfig:
    def __init__(self):
        # Financial Cognition Parameters
        self.financial = {
            'encoding': {
                'input_dim': 3,
                'hidden_dim': 64,
                'output_dim': 4
            },
            'memory': {
                'capacity': 1000,
                'retention': 0.9,
                'consolidation_interval': 1000
            },
            'state_space': {
                'input_dim': 4,
                'hidden_dim': 32,
                'num_layers': 1
            },
            'loss_weights': {
                'prediction': 0.4,
                'consistency': 0.3,
                'attention_stability': 0.1,
                'sparsity': 0.1,
                'coherence': 0.1
            }
        }
        
        # Core Consciousness Parameters (MX250 Optimized)
        self.core = {
            'dim': 128,          # Reduced from 256
            'max_depth': 3,      # Reduced from 5
            'adaptive_depth': True,
            'financial_projection_dim': 4,
            'batch_size': 4,     # Reduced from 32
            'use_amp': True      # Mixed Precision enabled
        }
        
        # Perception Parameters
        self.perception = {
            'cross_attention_dim': 128,  # Reduced from 256
            'financial_attention_heads': 2
        }

    def update(self, custom_config):
        """Deep update configuration with custom values"""
        for key, value in custom_config.items():
            if isinstance(value, dict):
                if key == 'MX250Config':
                    # Handle MX250 specific updates
                    for subkey, subvalue in value.items():
                        if subkey in self.__dict__:
                            self.__dict__[subkey].update(subvalue)
                else:
                    self.__dict__[key].update(value)
            else:
                self.__dict__[key] = value

class MX250Config:
    """Hardware-specific configuration for 2GB VRAM systems"""
    def __init__(self):
        self.core = {
            'dim': 128,
            'max_depth': 3,
            'batch_size': 4,
            'use_amp': True,
            'grad_accumulation': 2
        }
        self.financial = {
            'encoding': {'hidden_dim': 32}  # Reduced from 64
        }
        
    def apply(self, model):
        """Directly configure model parameters"""
        model.hidden_dim = self.core['dim']
        model.max_depth = self.core['max_depth']
        model.financial_proj = nn.Linear(4, self.core['dim'])
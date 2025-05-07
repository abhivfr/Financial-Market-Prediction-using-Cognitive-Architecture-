#!/usr/bin/env python
# cognitive.py - Cognitive architecture for financial forecasting

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TemporalEncoder(nn.Module):
    """Temporal encoder for financial sequence data"""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Encoded temporal representation [batch_size, seq_len, hidden_dim]
        """
        # LSTM processing
        outputs, (hidden, cell) = self.lstm(x)
        
        # Apply normalization and dropout
        outputs = self.norm(outputs)
        outputs = self.dropout(outputs)
        
        return outputs, hidden

class FinancialEncoder(nn.Module):
    """Encoder for financial features"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(FinancialEncoder, self).__init__()
        
        # Fully connected layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i == num_layers - 1 else (hidden_dim + current_dim) // 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Encoded financial representation [batch_size, hidden_dim]
        """
        return self.encoder(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention for financial data"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For visualization
        self.attention_weights = None
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len_q, hidden_dim]
            key: Key tensor [batch_size, seq_len_k, hidden_dim]
            value: Value tensor [batch_size, seq_len_v, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, seq_len_q, hidden_dim]
        """
        batch_size = query.size(0)
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights.detach()
        
        # Compute weighted values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def get_attention_weights(self):
        """Return stored attention weights for visualization"""
        return self.attention_weights

class FinancialMemory(nn.Module):
    """Financial memory module for storing and retrieving market patterns"""
    def __init__(self, input_dim, memory_dim, num_slots=50, dropout=0.1):
        super(FinancialMemory, self).__init__()
        
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        
        # Memory bank
        self.memory = nn.Parameter(torch.randn(num_slots, memory_dim))
        
        # Memory key/value networks
        self.key_network = nn.Linear(input_dim, memory_dim)
        self.value_network = nn.Linear(input_dim, memory_dim)
        
        # Output projection
        self.output_proj = nn.Linear(memory_dim, input_dim)
        
        # Memory usage tracker
        self.register_buffer('usage', torch.zeros(num_slots))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Memory access statistics
        self.access_count = 0
        self.write_count = 0
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Memory-enhanced output [batch_size, seq_len, features]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to keys and values
        keys = self.key_network(x)  # [batch_size, seq_len, memory_dim]
        values = self.value_network(x)  # [batch_size, seq_len, memory_dim]
        
        # Reshape for batch processing
        keys = keys.view(batch_size * seq_len, -1)
        values = values.view(batch_size * seq_len, -1)
        
        # Calculate similarity with memory
        similarity = F.cosine_similarity(
            keys.unsqueeze(1).expand(-1, self.num_slots, -1),
            self.memory.unsqueeze(0).expand(batch_size * seq_len, -1, -1),
            dim=2
        )  # [batch_size * seq_len, num_slots]
        
        # Apply temperature for sharper focus
        similarity = similarity / 0.5
        
        # Get attention weights
        attn_weights = F.softmax(similarity, dim=1)  # [batch_size * seq_len, num_slots]
        
        # Read from memory
        read_values = torch.matmul(attn_weights, self.memory)  # [batch_size * seq_len, memory_dim]
        
        # Update usage statistics
        with torch.no_grad():
            self.usage = 0.99 * self.usage + 0.01 * attn_weights.sum(dim=0)
            self.access_count += 1
        
        # Reshape read values
        read_values = read_values.view(batch_size, seq_len, -1)
        
        # Project to output space
        output = self.output_proj(read_values)
        
        # Combine with input via residual connection
        output = x + self.dropout(output)
        
        return output
    
    def consolidate_financial_memory(self, force=False):
        """
        Consolidate memory by removing least used slots and adding new ones
        
        Args:
            force: Force consolidation
        """
        if not force and self.access_count < 100:
            return
        
        with torch.no_grad():
            # Find least used slots
            num_to_replace = max(1, self.num_slots // 20)  # Replace ~5% of slots
            _, indices = torch.topk(self.usage, self.num_slots - num_to_replace, largest=True)
            mask = torch.ones_like(self.usage, dtype=torch.bool)
            mask[indices] = False
            
            # Initialize new slots with random values
            self.memory.data[mask] = torch.randn_like(self.memory.data[mask])
            
            # Reset usage for new slots
            self.usage[mask] = torch.ones_like(self.usage[mask]) * 0.01
            
            # Normalize memory for better learning
            self.memory.data = F.normalize(self.memory.data, dim=1)
            
            self.write_count += num_to_replace
    
    def get_memory_usage(self):
        """Get memory usage statistics"""
        usage_std = torch.std(self.usage).item()
        usage_entropy = -torch.sum(
            self.usage * torch.log(self.usage + 1e-10)
        ).item() / math.log(self.num_slots)
        
        return usage_entropy

class CrossDimensionalAttention(nn.Module):
    """Attention module for cross-dimensional interactions"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(CrossDimensionalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Cross-dimensional projections
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features, static_features):
        """
        Forward pass
        
        Args:
            temporal_features: Temporal features [batch_size, seq_len, hidden_dim]
            static_features: Static features [batch_size, hidden_dim]
            
        Returns:
            Cross-dimensional features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = temporal_features.size()
        
        # Project features
        time_proj = self.time_proj(temporal_features)
        feature_proj = self.feature_proj(static_features).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Cross-dimensional attention weights
        attention_scores = torch.bmm(
            time_proj,
            feature_proj.transpose(1, 2)
        ) / math.sqrt(self.hidden_dim)
        
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Apply attention
        attended_features = torch.bmm(attention_weights, feature_proj)
        
        # Combine with temporal features via residual connection
        output = temporal_features + self.dropout(attended_features)
        output = self.norm1(output)
        
        # Final projection
        projected = self.output_proj(output)
        output = output + self.dropout(projected)
        output = self.norm2(output)
        
        return output

class TemporalHierarchy(nn.Module):
    """Temporal hierarchy for multi-scale time processing"""
    def __init__(self, input_dim, hidden_dim, num_scales=3, dropout=0.1):
        super(TemporalHierarchy, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            TemporalEncoder(input_dim, hidden_dim, num_layers=1+i, dropout=dropout)
            for i in range(num_scales)
        ])
        
        # Scale fusion layer
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Multi-scale temporal features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Process at different scales
        scale_outputs = []
        
        for i, encoder in enumerate(self.scale_encoders):
            # Downsample for different scales
            scale_factor = 2 ** i
            if scale_factor > 1:
                # Use average pooling for downsampling
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=1,
                    padding=scale_factor // 2
                ).transpose(1, 2)
                
                # Ensure same sequence length
                x_scaled = x_scaled[:, :seq_len, :]
            else:
                x_scaled = x
            
            # Encode at this scale
            output, _ = encoder(x_scaled)
            
            # Ensure output has consistent dimensions
            output = output[:, :seq_len, :]
            
            scale_outputs.append(output)
        
        # Concatenate scale outputs
        multi_scale = torch.cat(scale_outputs, dim=2)
        
        # Fuse scales
        fused = self.fusion(multi_scale)
        
        # Apply normalization and dropout
        output = self.norm(fused)
        output = self.dropout(output)
        
        return output

class FusionNetwork(nn.Module):
    """Network for fusing different information sources"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FusionNetwork, self).__init__()
        
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *inputs):
        """
        Forward pass
        
        Args:
            *inputs: Variable number of input tensors [batch_size, ..., features]
            
        Returns:
            Fused representation [batch_size, ..., hidden_dim]
        """
        # Ensure all inputs have same batch size and sequence length
        batch_size = inputs[0].size(0)
        seq_len = inputs[0].size(1) if inputs[0].dim() > 2 else 1
        
        # Reshape all inputs to [batch_size, seq_len, features]
        reshaped_inputs = []
        total_features = 0
        
        for x in inputs:
            if x.dim() == 2:
                # [batch_size, features] -> [batch_size, 1, features]
                x = x.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Now x is [batch_size, seq_len, features]
            reshaped_inputs.append(x)
            total_features += x.size(2)
        
        # Concatenate along feature dimension
        combined = torch.cat(reshaped_inputs, dim=2)
        
        # Project to hidden dimension
        output = self.projection(combined)
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        return output

class MarketStateSequencePredictor(nn.Module):
    """Predictor for market state sequence"""
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=1, dropout=0.1):
        super(MarketStateSequencePredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.predictor = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Optional uncertainty head
        self.uncertainty_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Process with LSTM
        outputs, _ = self.lstm(x)
        outputs = self.dropout(outputs)
        
        # Get predictions
        predictions = self.predictor(outputs)
        
        # Estimate uncertainty (log variance)
        log_variance = self.uncertainty_head(outputs)
        uncertainty = torch.exp(log_variance)
        
        return {
            'market_state': predictions,
            'uncertainty': uncertainty
        }

class MetaController(nn.Module):
    """Meta-controller for managing cognitive processes"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(MetaController, self).__init__()
        
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # Control signals for attention, memory, prediction
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Control signals [batch_size, 3]
        """
        return self.controller(x)

class CognitiveArchitecture(nn.Module):
    """
    4D Cognitive Architecture for financial forecasting
    """
    def __init__(self, input_dim=5, hidden_dim=64, memory_size=50, output_dim=4, seq_length=20):
        """
        Initialize cognitive architecture
        
        Args:
            input_dim: Dimension of input features (internal representation)
            hidden_dim: Hidden dimension size
            memory_size: Number of memory slots
            output_dim: Dimension of output predictions
            seq_length: Input sequence length
        """
        super(CognitiveArchitecture, self).__init__()
        
        # Initialize dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        
        # Real data feature dimension (EnhancedFinancialDataLoader provides 7 features)
        self.data_feature_dim = 7
        
        # Add projection layers to map from data features to internal representation
        self.feature_projection = nn.Linear(self.data_feature_dim, input_dim)
        
        # Component 1: Financial Encoding
        # Use the original input_dim to match checkpoint dimensions
        self.financial_encoder = FinancialEncoder(input_dim, hidden_dim)
        
        # Component 2: Temporal Processing
        # Use the original input_dim to match checkpoint dimensions
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim)
        self.temporal_hierarchy = TemporalHierarchy(input_dim, hidden_dim)
        
        # Component 3: Multi-head Attention
        self.attention = MultiHeadAttention(hidden_dim)
        self.cross_dimensional_attention = CrossDimensionalAttention(hidden_dim)
        
        # Component 4: Financial Memory
        self.financial_memory = FinancialMemory(hidden_dim, hidden_dim, memory_size)
        self.memory_gate = nn.Linear(hidden_dim, 1)
        
        # Fusion network
        self.fusion = FusionNetwork(hidden_dim * 3, hidden_dim)
        
        # Meta-controller
        self.meta_controller = MetaController(hidden_dim)
        
        # Market state sequence predictor
        self.market_state_sequence_predictor = MarketStateSequencePredictor(
            hidden_dim, hidden_dim, output_dim
        )
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # 3 regime classes: bull, bear, sideways
            nn.LogSoftmax(dim=1)
        )
        
        # Initialize weights
        self._init_weights()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu"):
        """
        Create a model from a checkpoint, handling dimension mismatches
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Initialized model
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Extract config or infer from weights
        config = {}
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Infer input dimension
            if 'financial_encoder.encoder.0.weight' in state_dict:
                config['input_dim'] = state_dict['financial_encoder.encoder.0.weight'].shape[1]
            elif 'temporal_encoder.lstm.weight_ih_l0' in state_dict:
                config['input_dim'] = state_dict['temporal_encoder.lstm.weight_ih_l0'].shape[1]
            
            # Infer hidden dimension
            if 'temporal_encoder.lstm.weight_ih_l0' in state_dict:
                config['hidden_dim'] = state_dict['temporal_encoder.lstm.weight_ih_l0'].shape[0] // 4
            
            # Infer memory size
            if 'financial_memory.memory' in state_dict:
                config['memory_size'] = state_dict['financial_memory.memory'].shape[0]
            
            # Infer output dimension
            if 'market_state_sequence_predictor.predictor.weight' in state_dict:
                config['output_dim'] = state_dict['market_state_sequence_predictor.predictor.weight'].shape[0]
        
        # Create model with inferred config
        model = cls(**config)
        
        # Special handling for feature projection (new layer)
        excluded_keys = []
        if 'feature_projection.weight' not in state_dict:
            excluded_keys.append('feature_projection.weight')
            excluded_keys.append('feature_projection.bias')
            
        # Create a new state dict by excluding the projection layer
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in excluded_keys}
        
        # Load state dict with strict=False to allow missing keys
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Initialize the feature projection weights properly
        nn.init.xavier_uniform_(model.feature_projection.weight)
        nn.init.zeros_(model.feature_projection.bias)
        
        return model
    
    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    # Check if parameter has at least 2 dimensions before using xavier_uniform_
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        # For 1D tensors, use normal initialization instead
                        nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, financial_data=None, financial_seq=None, volume=None):
        """
        Forward pass
        
        Args:
            financial_data: Financial feature data [batch_size, features]
            financial_seq: Financial sequence data [batch_size, seq_len, features]
            volume: Optional volume data [batch_size, seq_len, 1]
            
        Returns:
            Dictionary of outputs
        """
        batch_size = financial_seq.size(0)
        
        # Project input features to match internal dimensions
        if financial_seq.shape[-1] != self.input_dim:
            # Project sequence data [batch_size, seq_len, 7] -> [batch_size, seq_len, input_dim]
            seq_len = financial_seq.size(1)
            # Use reshape instead of view to avoid contiguity issues
            financial_seq_reshaped = financial_seq.reshape(batch_size * seq_len, -1)
            financial_seq_projected = self.feature_projection(financial_seq_reshaped)
            financial_seq = financial_seq_projected.reshape(batch_size, seq_len, self.input_dim)
        
        # Process financial data if provided
        if financial_data is not None:
            # Project point data if needed [batch_size, 7] -> [batch_size, input_dim]
            if financial_data.shape[-1] != self.input_dim:
                financial_data = self.feature_projection(financial_data)
            financial_features = self.financial_encoder(financial_data)
        else:
            # Use the last step of sequence as financial data
            financial_features = self.financial_encoder(financial_seq[:, -1, :])
        
        # Process temporal sequence
        temporal_features, _ = self.temporal_encoder(financial_seq)
        
        # Process multi-scale temporal features
        multi_scale_features = self.temporal_hierarchy(financial_seq)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Apply cross-dimensional attention
        cross_dim_features = self.cross_dimensional_attention(
            temporal_features, financial_features
        )
        
        # Apply memory
        memory_features = self.financial_memory(attended_features)
        
        # Calculate memory gate
        memory_gate = torch.sigmoid(self.memory_gate(financial_features)).unsqueeze(1)
        
        # Fuse different representations
        fused_features = self.fusion(
            attended_features,
            cross_dim_features,
            memory_features
        )
        
        # Apply meta-controller
        if financial_features.dim() == 2:
            control_signals = self.meta_controller(financial_features)
        else:
            control_signals = self.meta_controller(financial_features.mean(dim=1))
        
        attention_signal = control_signals[:, 0].unsqueeze(1).unsqueeze(2)
        memory_signal = control_signals[:, 1].unsqueeze(1).unsqueeze(2)
        prediction_signal = control_signals[:, 2].unsqueeze(1).unsqueeze(2)
        
        # Apply control signals
        controlled_features = (
            attention_signal * attended_features + 
            memory_signal * memory_features +
            (1 - attention_signal - memory_signal) * cross_dim_features
        )
        
        # Get market state predictions
        market_outputs = self.market_state_sequence_predictor(controlled_features)
        
        # Get regime probabilities from financial features
        regime_logits = self.regime_classifier(financial_features)
        regime_probabilities = torch.exp(regime_logits)
        
        # Calculate attention variance (for monitoring)
        if attention_weights is not None:
            attention_variance = torch.var(attention_weights, dim=-1).mean().detach()
        else:
            attention_variance = torch.tensor(0.0, device=financial_features.device)
        
        # Return all outputs
        outputs = {
            'market_state': market_outputs['market_state'],
            'uncertainty': market_outputs['uncertainty'],
            'regime_probabilities': regime_probabilities,
            'attention_weights': attention_weights,
            'attention_variance': attention_variance,
            'memory_usage': self.financial_memory.get_memory_usage(),
            'control_signals': control_signals
        }
        
        return outputs

class RegimeDetector(nn.Module):
    """Detector for market regimes"""
    def __init__(self, input_dim, hidden_dim, num_regimes=3, window_size=20):
        super(RegimeDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        self.window_size = window_size
        
        # Temporal feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regime classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_regimes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Financial sequence data [batch_size, seq_len, features]
            
        Returns:
            Regime probabilities [batch_size, num_regimes]
        """
        # Ensure sequence is at least window_size
        seq_len = x.size(1)
        if seq_len < self.window_size:
            padding = torch.zeros(
                x.size(0), self.window_size - seq_len, x.size(2),
                device=x.device
            )
            x = torch.cat([padding, x], dim=1)
        
        # Use only the last window_size steps
        x = x[:, -self.window_size:, :]
        
        # Transpose for 1D convolution [batch, channels, length]
        x = x.transpose(1, 2)
        
        # Extract features
        features = self.feature_extractor(x).squeeze(-1)
        
        # Classify regime
        regime_logits = self.classifier(features)
        regime_probs = torch.exp(regime_logits)
        
        return regime_probs

class MarketStatePredictor(nn.Module):
    """Predictor for future market states"""
    def __init__(self, input_dim, hidden_dim, output_dim, forecast_horizon=5):
        super(MarketStatePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Prediction heads for each forecast step
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(forecast_horizon)
        ])
        
        # Uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(forecast_horizon)
        ])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequence [batch_size, seq_len, features]
            
        Returns:
            Dictionary with predictions and uncertainties for each forecast step
        """
        # Process sequence
        outputs, (hidden, _) = self.lstm(x)
        
        # Last hidden state
        last_hidden = outputs[:, -1, :]
        
        # Generate predictions for each forecast step
        predictions = []
        uncertainties = []
        
        for i in range(self.forecast_horizon):
            pred = self.prediction_heads[i](last_hidden)
            uncertainty = torch.exp(self.uncertainty_heads[i](last_hidden))  # Log variance to variance
            
            predictions.append(pred)
            uncertainties.append(uncertainty)
        
        # Stack predictions and uncertainties
        all_predictions = torch.stack(predictions, dim=1)  # [batch, horizon, output_dim]
        all_uncertainties = torch.stack(uncertainties, dim=1)  # [batch, horizon, output_dim]
        
        return {
            'predictions': all_predictions,
            'uncertainties': all_uncertainties
        }

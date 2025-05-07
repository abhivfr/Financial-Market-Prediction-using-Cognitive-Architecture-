import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np

class RiskManager(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 max_position_size: float = 1.0,
                 initial_capital: float = 100000.0,
                 max_drawdown: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_position_size = max_position_size
        self.initial_capital = initial_capital
        self.max_drawdown = max_drawdown
        
        # Position sizing network
        self.position_sizer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Market + Risk + Portfolio features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1 for position size
        )
        
        # Stop loss generator
        self.stop_loss_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Stop loss and take profit levels
            nn.Sigmoid()
        )
        
        # Portfolio allocation network
        self.portfolio_allocator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)  # Outputs sum to 1 for allocation
        )
        
        # Risk assessment network
        self.risk_assessor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Risk score, VaR, and Expected Shortfall
        )
        
    def compute_position_size(self,
                            market_features: torch.Tensor,
                            risk_metrics: torch.Tensor,
                            portfolio_state: torch.Tensor) -> torch.Tensor:
        """Compute optimal position size based on market conditions and risk."""
        combined = torch.cat([market_features, risk_metrics, portfolio_state], dim=-1)
        position_size = self.position_sizer(combined) * self.max_position_size
        return position_size
        
    def generate_stop_levels(self,
                           market_features: torch.Tensor,
                           position_size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate stop loss and take profit levels."""
        combined = torch.cat([market_features, position_size], dim=-1)
        levels = self.stop_loss_generator(combined)
        stop_loss = levels[..., 0:1] * 0.1  # Max 10% stop loss
        take_profit = levels[..., 1:2] * 0.2  # Max 20% take profit
        return stop_loss, take_profit
        
    def allocate_portfolio(self,
                         market_features: torch.Tensor,
                         risk_metrics: torch.Tensor) -> torch.Tensor:
        """Generate portfolio allocation weights."""
        combined = torch.cat([market_features, risk_metrics], dim=-1)
        weights = self.portfolio_allocator(combined)
        return weights
        
    def assess_risk(self,
                   market_features: torch.Tensor,
                   position_size: torch.Tensor,
                   portfolio_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various risk metrics."""
        combined = torch.cat([market_features, position_size, portfolio_state], dim=-1)
        risk_metrics = self.risk_assessor(combined)
        
        return {
            'risk_score': torch.sigmoid(risk_metrics[..., 0]),
            'value_at_risk': torch.exp(risk_metrics[..., 1]),
            'expected_shortfall': torch.exp(risk_metrics[..., 2])
        }
        
    def forward(self,
               market_features: torch.Tensor,
               portfolio_state: torch.Tensor,
               current_drawdown: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate complete risk management decision.
        
        Args:
            market_features: Encoded market state
            portfolio_state: Current portfolio status
            current_drawdown: Current drawdown level (optional)
            
        Returns:
            Dict containing position sizes, stop levels, and risk metrics
        """
        # Initial risk assessment
        risk_metrics = self.assess_risk(
            market_features,
            torch.zeros_like(portfolio_state[..., :1]),  # Initial zero position
            portfolio_state
        )
        
        # Compute position size
        position_size = self.compute_position_size(
            market_features,
            torch.stack([v for v in risk_metrics.values()], dim=-1),
            portfolio_state
        )
        
        # Apply drawdown control
        if current_drawdown is not None:
            drawdown_scalar = torch.max(
                torch.zeros_like(current_drawdown),
                1 - (current_drawdown / self.max_drawdown)
            )
            position_size = position_size * drawdown_scalar
        
        # Generate stop levels
        stop_loss, take_profit = self.generate_stop_levels(
            market_features,
            position_size
        )
        
        # Generate portfolio allocation
        allocation = self.allocate_portfolio(
            market_features,
            torch.stack([v for v in risk_metrics.values()], dim=-1)
        )
        
        # Final risk assessment with decided position size
        final_risk_metrics = self.assess_risk(
            market_features,
            position_size,
            portfolio_state
        )
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'portfolio_allocation': allocation,
            'risk_metrics': final_risk_metrics
        }

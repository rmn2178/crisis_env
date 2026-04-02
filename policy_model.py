"""
policy_model.py — Lightweight masked policy network for candidate OpenEnv actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyShape:
    state_dim: int
    action_dim: int


class PolicyNetwork(nn.Module):
    """
    Encodes the global observation and scores each valid candidate action.
    The output is a discrete distribution over the candidate list for the
    current step, so OpenEnv payloads remain external to the model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shape = PolicyShape(state_dim=state_dim, action_dim=action_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_tensor: [batch, state_dim] or [state_dim]
            action_tensor: [num_actions, action_dim]
        Returns:
            logits: [num_actions]
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        if action_tensor.dim() != 2:
            raise ValueError("action_tensor must have shape [num_actions, action_dim]")

        state_embedding = self.state_encoder(state_tensor).squeeze(0)
        action_embedding = self.action_encoder(action_tensor)
        repeated_state = state_embedding.unsqueeze(0).expand(action_embedding.size(0), -1)
        logits = self.policy_head(torch.cat([repeated_state, action_embedding], dim=-1))
        return logits.squeeze(-1)

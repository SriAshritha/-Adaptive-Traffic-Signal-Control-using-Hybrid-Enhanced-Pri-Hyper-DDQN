import math

import torch
import torch.nn as nn


class PriDDQNNet(nn.Module):
    """
    Lightweight 1D-conv feature extractor to keep the paper's hybrid-agent
    flavor while remaining compatible with the repository's compact states.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def power_decay_epsilon(episode: int, total_episodes: int, eps_start: float, eps_end: float) -> float:
    if total_episodes <= 1:
        return eps_end
    exponent_base = math.pow(eps_end / eps_start, 1.0 / max(total_episodes - 1, 1))
    return max(eps_end, eps_start * math.pow(exponent_base, episode))

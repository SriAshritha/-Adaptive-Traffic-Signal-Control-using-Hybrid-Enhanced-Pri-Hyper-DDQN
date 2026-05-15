import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphTemporalEncoder(nn.Module):
    def __init__(self, node_feat_dim: int = 2, embed_dim: int = 64):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, embed_dim)
        self.temporal = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        incidence = torch.tensor(
            [
                [1, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("incidence", incidence)

    def forward(self, state_seq: torch.Tensor):
        batch, time_steps, state_dim = state_seq.shape
        node_feats = state_seq[:, :, :16].reshape(batch, time_steps, 8, 2)
        x = self.node_proj(node_feats)

        h = self.incidence
        h_t = h.t()
        edge_norm = h.sum(dim=1, keepdim=True).clamp_min(1.0)
        node_norm = h_t.sum(dim=1, keepdim=True).clamp_min(1.0)

        edge_msg = torch.einsum("eh,btnd->bted", h / edge_norm, x)
        node_msg = torch.einsum("ne,bted->btnd", h_t / node_norm, edge_msg)
        x = x + node_msg

        x = x.permute(0, 2, 1, 3).reshape(batch * 8, time_steps, -1)
        _, hidden = self.temporal(x)
        node_embed = hidden[-1].reshape(batch, 8, -1)
        node_embed = self.out_proj(node_embed)

        global_embed = node_embed.mean(dim=1)
        return node_embed, global_embed


class MultiAgentActor(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = HypergraphTemporalEncoder(embed_dim=embed_dim)
        self.agent_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim * 2 + 1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(4)
            ]
        )
        self.action_groups = (
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
        )

    def forward(self, state_seq: torch.Tensor):
        node_embed, global_embed = self.encoder(state_seq)
        current_phase = state_seq[:, -1, -1:].float()
        logits = []
        for action_idx, head in enumerate(self.agent_heads):
            indices = self.action_groups[action_idx]
            local_embed = node_embed[:, indices, :].mean(dim=1)
            features = torch.cat([local_embed, global_embed, current_phase], dim=-1)
            logits.append(head(features))
        return torch.cat(logits, dim=-1), global_embed


class CentralizedCritic(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = HypergraphTemporalEncoder(embed_dim=embed_dim)
        self.q_net = nn.Sequential(
            nn.Linear(embed_dim + 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_seq: torch.Tensor, actions_one_hot: torch.Tensor):
        _, global_embed = self.encoder(state_seq)
        current_phase = state_seq[:, -1, -1:].float()
        x = torch.cat([global_embed, actions_one_hot, current_phase], dim=-1)
        return self.q_net(x)


def categorical_sample(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    log_prob = dist.log_prob(action).unsqueeze(-1)
    return action, log_prob, probs

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import RESULT_PATHS, TRAIN_CONFIG
from model import CentralizedCritic, MultiAgentActor, categorical_sample


class MASACAgent:
    def __init__(self, device: torch.device):
        self.device = device
        embed_dim = TRAIN_CONFIG["embedding_dim"]
        hidden_dim = TRAIN_CONFIG["hidden_dim"]
        self.actor = MultiAgentActor(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
        self.critic1 = CentralizedCritic(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
        self.critic2 = CentralizedCritic(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
        self.target_critic1 = CentralizedCritic(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
        self.target_critic2 = CentralizedCritic(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=TRAIN_CONFIG["actor_lr"])
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=TRAIN_CONFIG["critic_lr"])
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=TRAIN_CONFIG["critic_lr"])
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=TRAIN_CONFIG["alpha_lr"])
        self.target_entropy = -TRAIN_CONFIG["target_entropy_scale"] * 4.0
        self.gamma = TRAIN_CONFIG["gamma"]
        self.tau = TRAIN_CONFIG["tau"]

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state_seq, greedy: bool = False):
        state_seq = torch.tensor(state_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.actor(state_seq)
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.distributions.Categorical(probs=probs).sample()
        return int(action.item()), probs.squeeze(0).cpu().tolist()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions_one_hot = F.one_hot(actions, num_classes=4).float()

        with torch.no_grad():
            next_logits, _ = self.actor(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs.clamp_min(1e-8))
            q1_next = []
            q2_next = []
            for action_idx in range(4):
                one_hot = F.one_hot(
                    torch.full((states.size(0),), action_idx, device=self.device, dtype=torch.long),
                    num_classes=4,
                ).float()
                q1_next.append(self.target_critic1(next_states, one_hot))
                q2_next.append(self.target_critic2(next_states, one_hot))
            q1_next = torch.cat(q1_next, dim=1)
            q2_next = torch.cat(q2_next, dim=1)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (min_q_next - self.alpha.detach() * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1.0 - dones) * self.gamma * v_next

        q1 = self.critic1(states, actions_one_hot)
        q2 = self.critic2(states, actions_one_hot)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        logits, _ = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-8))
        q1_all = []
        q2_all = []
        for action_idx in range(4):
            one_hot = F.one_hot(
                torch.full((states.size(0),), action_idx, device=self.device, dtype=torch.long),
                num_classes=4,
            ).float()
            q1_all.append(self.critic1(states, one_hot))
            q2_all.append(self.critic2(states, one_hot))
        q1_all = torch.cat(q1_all, dim=1)
        q2_all = torch.cat(q2_all, dim=1)
        min_q = torch.min(q1_all, q2_all)
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(entropy.mean().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)

    def save(self):
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
        }
        torch.save(payload, RESULT_PATHS["checkpoint"])

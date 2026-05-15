import random

import numpy as np


class PriorityReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition, priority: float):
        priority = float(max(priority, 1e-6))
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        current_priorities = self.priorities[: len(self.buffer)]
        scaled = np.power(current_priorities, self.alpha)
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = np.power(len(self.buffer) * probs[indices], -beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(max(priority, 1e-6))

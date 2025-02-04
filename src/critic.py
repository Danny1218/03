import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)  # Linear layer for scoring

    def forward(self, hidden_states):
        # Average pooling over sequence dimension and apply linear layer
        return self.fc(hidden_states.mean(dim=1)) 
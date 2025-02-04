import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # Expanded critic with an extra linear layer and nonlinearity
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),  # Nonlinear activation for deeper representations
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        # Use mean pooling of hidden states
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)

if __name__ == '__main__':
    critic = Critic()
    sample = torch.randn(2, 10, 128)  # batch=2, sequence length=10, hidden_size=128
    output = critic(sample)
    print(output)
    assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
    print("Critic output validated.") 
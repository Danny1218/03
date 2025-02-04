import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)  # Linear layer for scoring

    def forward(self, hidden_states):
        # Average pooling over sequence dimension and apply linear layer
        return self.fc(hidden_states.mean(dim=1)) 

if __name__ == '__main__':
    critic = Critic()
    sample = torch.randn(2, 10, 128)  # batch=2, sequence length=10, hidden_size=128
    output = critic(sample)
    print(output)
    assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
    print("Critic output validated.") 
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.out = nn.Linear(hidden_size // 4, 1)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        x = self.activation(self.fc1(pooled))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

if __name__ == '__main__':
    critic = Critic()
    sample = torch.randn(2, 10, 128)  # batch=2, sequence length=10, hidden_size=128
    output = critic(sample)
    print(output)
    assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
    print("Critic output validated.") 
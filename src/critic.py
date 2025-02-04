import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # Enhanced two-layer architecture
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # average pooling
        x = torch.relu(self.fc1(pooled))
        return self.fc2(x)

    def compute_loss(self, predictions, targets):
        # Compute normalized MSE loss for critic training
        mse_loss = nn.MSELoss()
        normalized_preds = (predictions - predictions.mean()) / (predictions.std() + 1e-8)
        normalized_targets = (targets - targets.mean()) / (targets.std() + 1e-8)
        return mse_loss(normalized_preds, normalized_targets)

if __name__ == '__main__':
    critic = Critic()
    sample = torch.randn(2, 10, 128)  # batch=2, sequence length=10, hidden_size=128
    output = critic(sample)
    print(output)
    assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
    print("Critic output validated.") 
import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.register_buffer('baseline', torch.zeros(1))
        self.decay = 0.99

    def forward(self, hidden_states):
        avg_pool = hidden_states.mean(dim=1)
        max_pool, _ = hidden_states.max(dim=1)
        std_pool = hidden_states.std(dim=1)
        combined = (avg_pool + max_pool + std_pool) / 3
        raw_score = self.fc(combined)
        norm_score = raw_score - self.baseline
        if self.training:
            self.baseline.mul_(self.decay).add_((1 - self.decay) * raw_score.detach().mean())
        return norm_score

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
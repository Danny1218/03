import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super().__init__()
        # Use MultiheadAttention to refine hidden state representations
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        # A small MLP for final reward estimation
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        pooled = attn_output.mean(dim=1)  # aggregate over sequence length
        return self.fc(pooled)

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
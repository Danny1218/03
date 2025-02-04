import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, hidden_size=128, nhead=4, num_layers=1):
        super().__init__()
        # Enhanced critic using a transformer encoder for attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.register_buffer('baseline', torch.zeros(1))
        self.decay = 0.99

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        # transpose for transformer: [seq_len, batch, hidden_size]
        x = hidden_states.transpose(0, 1)
        x = self.transformer(x)
        # Aggregate information via average pooling
        x = x.mean(dim=0)  
        raw_score = self.fc(x)
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
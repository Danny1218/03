import torch
import torch.nn as nn

class SelectiveMoE(nn.Module):
    def __init__(self, in_features, out_features, num_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        gate_scores = self.gate(x)
        gate_weights = torch.softmax(gate_scores, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return output 
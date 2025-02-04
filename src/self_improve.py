from torch.optim import Adam
from transformer_model import model
from critic import Critic
import torch  # added for self_improve function

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)

# Generate candidate responses from a prompt using model.generate
def generate_candidates(prompt, num_candidates=5):
    model.eval()
    candidates = [model.generate(prompt, max_length=50, do_sample=True) for _ in range(num_candidates)]
    model.train()
    return candidates

def self_improve(prompt, num_candidates=5):
    cands = generate_candidates(prompt, num_candidates)
    scores = torch.stack([critic(model(c, output_hidden_states=True).hidden_states[-1]).mean() for c in cands])
    best = cands[torch.argmax(scores)]
    loss = -scores.max()
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    return best

if __name__ == '__main__':
    print('Model Optimizer:', optimizer_model)
    print('Critic Optimizer:', optimizer_critic) 
from torch.optim import Adam
from transformer_model import model
from critic import Critic

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)

# Generate candidate responses from a prompt using model.generate
def generate_candidates(prompt, num_candidates=5):
    model.eval()
    candidates = [model.generate(prompt, max_length=50, do_sample=True) for _ in range(num_candidates)]
    model.train()
    return candidates

if __name__ == '__main__':
    print('Model Optimizer:', optimizer_model)
    print('Critic Optimizer:', optimizer_critic) 
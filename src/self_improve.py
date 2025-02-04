from torch.optim import Adam
from transformer_model import model
from critic import Critic

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)

if __name__ == '__main__':
    print('Model Optimizer:', optimizer_model)
    print('Critic Optimizer:', optimizer_critic) 
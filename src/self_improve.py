from torch.optim import Adam
from transformer_model import model
from critic import Critic
import torch  # added for self_improve function
import torch.nn.functional as F  # added for critic update

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)  # added critic optimizer

# Task 26: Self-consistency via candidate sampling and Task 31: Selecting best candidate with highest reward
def self_improve(prompt, num_candidates=5):
    model.eval()
    with torch.no_grad():
        cands = [model.generate(prompt, max_length=50, do_sample=True) for _ in range(num_candidates)]
    model.train()
    # Compute rewards using critic on extracted hidden states
    rewards = torch.stack([critic(model(c, output_hidden_states=True).hidden_states[-1]).mean() for c in cands])
    best_idx = torch.argmax(rewards)  # select index with the highest reward
    best = cands[best_idx]
    loss = -rewards[best_idx]  # RL update: negative reward loss
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    # Critic update mechanism (Task 35): use MSE loss against a baseline target of 1.0
    bh = model(best, output_hidden_states=True).hidden_states[-1]
    target = torch.ones(1, device=bh.device)
    optimizer_critic.zero_grad()
    closs = F.mse_loss(critic(bh).mean(), target)
    closs.backward()
    optimizer_critic.step()
    return best

if __name__ == '__main__':
    import sys
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = tokenizer.encode(prompt_text, return_tensors='pt')
    improved = self_improve(prompt)
    print(tokenizer.decode(improved[0], skip_special_tokens=True)) 
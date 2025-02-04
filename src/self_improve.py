from torch.optim import Adam
from transformer_model import model
from critic import Critic
import torch  # added for self_improve function
import torch.nn.functional as F  # added for critic update

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)  # added critic optimizer

# Added function to update critic parameters based on evaluation (Task 36)
def update_critic(cand):
    bh = model(cand, output_hidden_states=True).hidden_states[-1]
    target = torch.ones(1, device=bh.device)  # baseline target
    loss = F.mse_loss(critic(bh).mean(), target)
    optimizer_critic.zero_grad()
    loss.backward()
    optimizer_critic.step()

# Task 38: Integrate reinforcement learning update steps
def self_improve(prompt, num_candidates=5):
    model.eval()
    with torch.no_grad():
        cands = [model.generate(prompt, max_length=50, do_sample=True) for _ in range(num_candidates)]
    model.train()
    rewards = torch.stack([critic(model(c, output_hidden_states=True).hidden_states[-1]).mean() for c in cands])
    best_index = rewards.argmax().item()
    best_reward = rewards[best_index]
    best_candidate = cands[best_index]

    # Compute teacher-forcing loss on best candidate weighted by its critic reward
    outputs = model(best_candidate, labels=best_candidate)
    loss_rl = best_reward * outputs.loss
    optimizer_model.zero_grad()
    loss_rl.backward()
    optimizer_model.step()

    update_critic(best_candidate)
    return best_candidate

if __name__ == '__main__':
    import sys
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = tokenizer.encode(prompt_text, return_tensors='pt')
    improved = self_improve(prompt)
    print(tokenizer.decode(improved[0], skip_special_tokens=True)) 
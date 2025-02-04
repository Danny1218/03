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

# Task 38-39: Integrated candidate evaluation within a loop
def self_improve(prompt, num_candidates=5):
    model.eval()
    candidates, rewards = [], []
    for _ in range(num_candidates):
        with torch.no_grad():
            cand = model.generate(prompt, max_length=50, do_sample=True)
        out = model(cand, output_hidden_states=True)
        candidates.append(cand)
        rewards.append(critic(out.hidden_states[-1]).mean())
    best_index = torch.argmax(torch.stack(rewards)).item()
    best_candidate = candidates[best_index]

    model.train()
    outputs = model(best_candidate, labels=best_candidate)
    loss_rl = rewards[best_index] * outputs.loss
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
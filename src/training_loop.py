import logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import GPT2Tokenizer

from src.transformer_model import model
from src.critic import Critic
from src.config import LEARNING_RATE, NUM_CANDIDATES, MAX_NEW_TOKENS, MCTS_SIMS, MODEL_NAME

# Setup logging and tokenizer
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Move model to device
model.to(device)

# Initialize critic and move to device
critic = Critic()
critic.to(device)

def preprocess(text):
    tokens = _tokenizer(text, return_tensors='pt')
    return {k: v.to(device) for k, v in tokens.items()}

# Initialize optimizers
optimizer_model = Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_critic = Adam(critic.parameters(), lr=LEARNING_RATE)

def update_critic(cand):
    bh = model(cand, output_hidden_states=True).hidden_states[-1]
    target = torch.ones(1, device=bh.device)  # baseline target
    loss = F.mse_loss(critic(bh).mean(), target)
    optimizer_critic.zero_grad(); loss.backward(); optimizer_critic.step()


def mcts_expand(node, sims=MCTS_SIMS):
    children = [model.generate(node, attention_mask=torch.ones_like(node), max_new_tokens=MAX_NEW_TOKENS, do_sample=True, pad_token_id=_tokenizer.eos_token_id) for _ in range(sims)]
    rewards = [critic(model(child, output_hidden_states=True).hidden_states[-1]).mean() for child in children]
    best_index = torch.argmax(torch.stack(rewards))
    return children[best_index], rewards[best_index]


def self_improve(prompt):
    model.eval(); candidates, rewards = [], []
    for _ in range(NUM_CANDIDATES):
        try:
            with torch.no_grad():
                cand = model.generate(prompt['input_ids'], attention_mask=prompt['attention_mask'], max_new_tokens=MAX_NEW_TOKENS, do_sample=True, pad_token_id=_tokenizer.eos_token_id)
            if cand is None: raise ValueError('Generation returned None')
        except Exception as e:
            logging.error('Generation failed: %s', e); continue
        out = model(cand, output_hidden_states=True)
        candidates.append(cand); rewards.append(critic(out.hidden_states[-1]).mean())
    if not candidates: raise RuntimeError('No valid candidates generated')
    for i, cand in enumerate(candidates):
        logging.info('Candidate %d: %s, reward: %.4f', i, _tokenizer.decode(cand[0], skip_special_tokens=True), rewards[i].item())
    best_index = torch.argmax(torch.stack(rewards)).item()
    logging.info('Selected candidate %d with reward: %.4f', best_index, rewards[best_index].item())
    best_candidate = candidates[best_index]
    best_candidate, mcts_r = mcts_expand(best_candidate)
    if mcts_r is not None: logging.info('MCTS updated candidate reward: %.4f', mcts_r.item())
    final_reward = mcts_r if mcts_r is not None else rewards[best_index]
    model.train()
    outputs = model(best_candidate, labels=best_candidate)
    loss_rl = final_reward * outputs.loss
    optimizer_model.zero_grad(); loss_rl.backward(); optimizer_model.step()
    update_critic(best_candidate)
    return best_candidate

if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 
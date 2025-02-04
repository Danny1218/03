if __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import logging
import logging.config
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import GPT2Tokenizer

from src.transformer_model import model
from src.critic import Critic
from src.config import LEARNING_RATE, NUM_CANDIDATES, MAX_NEW_TOKENS, MCTS_SIMS, MODEL_NAME, LOG_LEVEL, LOG_FILE

# Setup logging and tokenizer
logging_config = {
    'version': 1,
    'formatters': {
         'default': {
              'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
         }
    },
    'handlers': {
         'console': {
              'class': 'logging.StreamHandler',
              'formatter': 'default',
              'level': LOG_LEVEL
         },
         'file': {
              'class': 'logging.FileHandler',
              'formatter': 'default',
              'filename': LOG_FILE,
              'level': LOG_LEVEL
         }
    },
    'root': {
         'handlers': ['console', 'file'],
         'level': LOG_LEVEL
    }
}
logging.config.dictConfig(logging_config)
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
    target = torch.tensor(1.0, device=bh.device)  # baseline target as a scalar
    loss = F.mse_loss(critic(bh).mean(), target)
    optimizer_critic.zero_grad()
    loss.backward()
    optimizer_critic.step()

# Added checkpoint saving function

def save_checkpoint(step=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_model_state": optimizer_model.state_dict(),
        "optimizer_critic_state": optimizer_critic.state_dict(),
    }
    fname = f"checkpoint_{step}.pt" if step is not None else "checkpoint.pt"
    torch.save(checkpoint, fname)


def mcts_expand(node, sims=MCTS_SIMS):
    children = [model.generate(node, attention_mask=torch.ones_like(node), max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_k=50, top_p=0.95, pad_token_id=_tokenizer.eos_token_id) for _ in range(sims)]
    rewards = [critic(model(child, output_hidden_states=True).hidden_states[-1]).mean() for child in children]
    best_index = torch.argmax(torch.stack(rewards))
    return children[best_index], rewards[best_index]

# Added helper functions to modularize candidate generation and evaluation

def generate_candidates(prompt):
    candidates = []
    for i in range(NUM_CANDIDATES):
        try:
            with torch.no_grad():
                cand = model.generate(prompt['input_ids'],
                                      attention_mask=prompt['attention_mask'],
                                      max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=True,
                                      pad_token_id=_tokenizer.eos_token_id)
            if cand is None or cand.numel() == 0:
                raise ValueError('Generation returned empty candidate')
            candidates.append(cand)
        except Exception as e:
            logging.error('Candidate generation failed at attempt %d: %s', i, e)
    if not candidates:
        raise RuntimeError('No valid candidates generated')
    return candidates


def evaluate_candidate(cand):
    try:
        out = model(cand, output_hidden_states=True)
        return critic(out.hidden_states[-1]).mean()
    except Exception as e:
        logging.error('Evaluation failed: %s', e)
        return None

# Updated self_improve function incorporating modular candidate generation and evaluation

def self_improve(prompt):
    model.eval()
    candidates = generate_candidates(prompt)
    rewards = []
    for cand in candidates:
        reward = evaluate_candidate(cand)
        if reward is not None:
            rewards.append(reward)
    if not rewards:
        raise RuntimeError('No valid candidates evaluated')
    avg_reward = torch.stack(rewards).mean()
    logging.info("Metrics: Average Candidate Reward: %.4f", avg_reward.item())
    for i, cand in enumerate(candidates):
        try:
            decoded = _tokenizer.decode(cand[0], skip_special_tokens=True)
        except Exception as e:
            decoded = '<Decoding failed>'
            logging.error('Decoding candidate %d failed: %s', i, e)
        logging.info('Candidate %d: %s, reward: %.4f', i, decoded, rewards[i].item())
    best_index = torch.argmax(torch.stack(rewards)).item()
    logging.info('Selected candidate %d with reward: %.4f', best_index, rewards[best_index].item())
    best_candidate = candidates[best_index]
    try:
        best_candidate, mcts_r = mcts_expand(best_candidate)
        if mcts_r is not None:
            logging.info('MCTS updated candidate reward: %.4f', mcts_r.item())
    except Exception as e:
        logging.error('MCTS expansion failed: %s', e)
        mcts_r = None
    final_reward = mcts_r if mcts_r is not None else rewards[best_index]
    model.train()
    outputs = model(best_candidate, labels=best_candidate)
    logging.info("Metrics: Loss (before RL update): %.4f", outputs.loss.item())
    loss_rl = final_reward * outputs.loss
    logging.info("Metrics: RL Loss: %.4f", loss_rl.item())
    logging.info("Metrics: Perplexity: %.4f", torch.exp(outputs.loss).item())
    optimizer_model.zero_grad()
    loss_rl.backward()
    optimizer_model.step()
    update_critic(best_candidate)
    save_checkpoint()
    return best_candidate

if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 
import logging
import time
import torch

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from src.training_loop import preprocess, _tokenizer, model, critic, optimizer_model
else:
    from .training_loop import preprocess, _tokenizer, model, critic, optimizer_model


def self_improve(prompt, num_candidates=5):
    logger.info("Starting self_improve with %d candidates", num_candidates)
    candidates = []
    rewards = []
    
    # Candidate generation with error handling and timing
    for i in range(num_candidates):
        start = time.time()
        try:
            outputs = model.generate(prompt, max_length=50, do_sample=True)
            candidates.append(outputs)
            logger.info("Candidate %d generated in %.4f sec", i, time.time()-start)
        except Exception as e:
            logger.error("Candidate %d generation failed: %s", i, e)
    
    model.train()
    
    # Evaluate candidates with error handling and timing
    for i, cand in enumerate(candidates):
        start_eval = time.time()
        try:
            outputs = model(cand, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            reward = critic(hidden)
            rewards.append(reward)
            logger.info("Candidate %d evaluated in %.4f sec", i, time.time()-start_eval)
        except Exception as e:
            logger.error("Candidate %d evaluation failed: %s", i, e)
            rewards.append(torch.tensor(0.0))
            
    # Log candidate diversity
    try:
        reward_vals = [r.item() for r in rewards]
        diversity = max(reward_vals) - min(reward_vals)
        logger.info("Candidate rewards: %s | Diversity: %.4f", reward_vals, diversity)
    except Exception as e:
        logger.error("Error computing candidate diversity: %s", e)
    
    # Select best candidate
    try:
        best_index = torch.argmax(torch.stack(rewards))
        best_candidate = candidates[best_index]
    except Exception as e:
        logger.error("Selection of best candidate failed: %s", e)
        best_candidate = None
    
    # RL update step with error handling
    try:
        loss = -rewards[best_index].mean()
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        logger.info("RL update completed with loss: %.4f", loss.item())
    except Exception as e:
        logger.error("RL update failed: %s", e)
    
    return best_candidate


if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 
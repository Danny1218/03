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


def retrieve_knowledge_context(prompt):
    # Minimal stub for external retrieval.
    # In production, replace with API call or knowledge base query.
    return "External context: Recent AI advances show improved factual grounding with retrieval augmentation."


# Added helper function for extracting final answer from chain-of-thought generation
def extract_final_answer(generated_text: str) -> str:
    # If 'Final Answer:' marker exists, extract text after it; otherwise, return full text
    if 'Final Answer:' in generated_text:
        return generated_text.split('Final Answer:')[-1].strip()
    return generated_text.strip()


def self_improve(prompt, num_candidates=5):
    logger.info("Starting self_improve with %d candidates", num_candidates)
    model.eval()
    candidates = []
    
    # Generate candidates with chain-of-thought reasoning
    for _ in range(num_candidates):
        # Generate candidate with chain-of-thought reasoning
        input_prompt = prompt + " Let's think step by step:"
        outputs = model.generate(input_prompt, max_length=100, do_sample=True)
        candidate = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        candidates.append(candidate)
    
    model.train()
    
    # Voting system: select candidate with highest pairwise word overlap
    def vote_score(text):
        words = set(text.split())
        return sum(len(words & set(other.split())) for other in candidates if other != text)

    best_candidate = max(candidates, key=vote_score)
    
    return best_candidate


if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {improved}") 
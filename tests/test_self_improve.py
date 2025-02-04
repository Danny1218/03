import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import GPT2Tokenizer
from src.self_improve import self_improve, preprocess, _tokenizer


def test_self_improve():
    torch.manual_seed(42)  # set seed for reproducibility
    prompt = preprocess('Test prompt')
    result = self_improve(prompt, num_candidates=2)
    decoded = _tokenizer.decode(result[0], skip_special_tokens=True)
    assert decoded.strip(), 'self_improve should return non-empty output'


if __name__ == '__main__':
    test_self_improve()
    print('Test passed.') 
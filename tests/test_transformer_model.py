import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
from src.transformer_model import model
from transformers import GPT2LMHeadModel

class TestTransformerModel(unittest.TestCase):
    def test_instance(self):
        self.assertIsInstance(model, GPT2LMHeadModel)
    def test_forward(self):
        cfg = model.config
        x = torch.randint(0, cfg.vocab_size, (1, 10))
        out = model(x, labels=x)
        self.assertIn('loss', out)
        self.assertEqual(out.logits.shape, (1, 10, cfg.vocab_size))

if __name__ == '__main__':
    unittest.main() 
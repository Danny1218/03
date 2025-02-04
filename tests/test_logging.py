import os
import shutil
import torch
import unittest
from src.training_loop import self_improve

class TestTensorBoardLogging(unittest.TestCase):
    def setUp(self):
        # Clean log directory if it exists
        if os.path.exists('runs/tensorboard_logs'):
            shutil.rmtree('runs/tensorboard_logs')

    def test_tensorboard_logs_created(self):
        prompt = torch.tensor([[101, 102, 103]])
        _ = self_improve(prompt)
        # Check if log directory exists and is not empty
        self.assertTrue(os.path.exists('runs/tensorboard_logs'))
        logs = os.listdir('runs/tensorboard_logs')
        self.assertTrue(len(logs) > 0)

if __name__ == '__main__':
    unittest.main() 
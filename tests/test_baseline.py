import os
import unittest

import torch
import torch.nn.functional as F

from agents import AgentConfig
from baseline import Baseline


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig()

    def test_baseline_sender(self):
        batch_size = 3
        baseline = Baseline(self.config, 'sender')

        img = torch.FloatTensor(batch_size, self.config.image_in)
        msg = torch.FloatTensor(batch_size, self.config.message_in)
        out = baseline(img, msg, None)

        self.assertEqual(out.size(), (batch_size, 1))

    def test_baseline_receiver(self):
        batch_size = 3
        baseline = Baseline(self.config, 'receiver')

        msg = torch.FloatTensor(batch_size, self.config.message_in)
        state = torch.FloatTensor(batch_size, self.config.receiver_hidden_dim)
        out = baseline(None, msg, state)

        self.assertEqual(out.size(), (batch_size, 1))


if __name__ == '__main__':
    unittest.main()

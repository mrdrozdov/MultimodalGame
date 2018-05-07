import os
import unittest

import torch

from agents import AgentConfig, Sender


class TestSender(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig()

    def test_forward_train(self):
        batch_size = 3
        sender = Sender(self.config)
        sender.train()

        msg = torch.FloatTensor(batch_size, self.config.message_in)
        img = torch.FloatTensor(batch_size, self.config.image_in)
        out_msg, dist = sender(msg, None, img, None)

        self.assertEqual(dist.le(1).sum().item(), dist.numel())
        self.assertEqual(dist.ge(0).sum().item(), dist.numel())
        self.assertEqual(out_msg.eq(0).sum().item() + out_msg.eq(1).sum(), out_msg.numel())

    def test_forward_eval(self):
        batch_size = 3
        sender = Sender(self.config)
        sender.eval()

        msg = torch.FloatTensor(batch_size, self.config.message_in)
        img = torch.FloatTensor(batch_size, self.config.image_in)
        out_msg, dist = sender(msg, None, img, None)

        self.assertEqual(dist.le(1).sum().item(), dist.numel())
        self.assertEqual(dist.ge(0).sum().item(), dist.numel())
        self.assertEqual(out_msg.eq(0).sum().item() + out_msg.eq(1).sum(), out_msg.numel())


if __name__ == '__main__':
    unittest.main()

import os
import unittest

import torch
import torch.nn.functional as F

from agents import AgentConfig, Sender, Receiver
from exchange import Exchange


class TestExchange(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig()

    def test_exchange(self):
        batch_size = 3
        length = 5
        descriptors = torch.FloatTensor(self.config.nclasses, self.config.descriptor_dim).normal_()
        sender = Sender(self.config)
        sender.eval()
        receiver = Receiver(self.config)
        receiver.eval()
        exchange = Exchange(sender, receiver, descriptors)

        image = torch.FloatTensor(batch_size, self.config.image_in).normal_()
        results = exchange.exchange(image, length)

        self.assertEqual(len(results), length)


if __name__ == '__main__':
    unittest.main()

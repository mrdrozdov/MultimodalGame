import os
import unittest

import torch
import torch.nn.functional as F

from agents import AgentConfig, Sender, Receiver
from baseline import Baseline
from exchange import Exchange, ExchangeModel
from trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig()

    def test_trainer(self):
        batch_size = 3
        length = 5
        descriptors = torch.FloatTensor(self.config.nclasses, self.config.descriptor_dim).normal_()
        sender = Sender(self.config)
        sender.eval()
        receiver = Receiver(self.config)
        receiver.eval()
        exchange_model = ExchangeModel(self.config)
        baseline_sender = Baseline(self.config, 'sender')
        baseline_receiver = Baseline(self.config, 'receiver')
        exchange = Exchange(exchange_model, sender, receiver, baseline_sender, baseline_receiver, descriptors)
        trainer = Trainer(exchange)


        image = torch.FloatTensor(batch_size, self.config.image_in).normal_()
        target_dist = F.softmax(torch.FloatTensor(batch_size, self.config.nclasses).normal_(), dim=1)
        target = target_dist.argmax(dim=1)
        trainer_loss = trainer.run_step(image, target)

        self.assertEqual(trainer_loss.sender_message_loss.numel(), 1)
        self.assertEqual(trainer_loss.receiver_message_loss.numel(), 1)
        self.assertEqual(trainer_loss.stop_loss.numel(), 1)
        self.assertEqual(trainer_loss.baseline_loss_sender.numel(), 1)
        self.assertEqual(trainer_loss.baseline_loss_receiver.numel(), 1)
        self.assertEqual(trainer_loss.xent_loss.numel(), 1)


if __name__ == '__main__':
    unittest.main()

import os
import unittest

import torch
import torch.nn.functional as F

from agents import AgentConfig, Sender, Receiver


def check_bernoulli_out(self, out):
    self.assertEqual(out.eq(0).sum().item() + out.eq(1).sum(), out.numel())


def check_bernoulli_dist(self, dist):
    self.assertTrue(dist.le(1).sum().item() == dist.numel() \
        and dist.ge(0).sum().item(), dist.numel())


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


class TestReceiver(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig()

    def test_forward_train(self):
        batch_size = 3
        receiver = Receiver(self.config)
        receiver.train()

        msg = torch.FloatTensor(batch_size, self.config.message_in).normal_()
        state = torch.FloatTensor(batch_size, self.config.receiver_hidden_dim).fill_(0)
        descriptors = torch.FloatTensor(self.config.nclasses, self.config.descriptor_dim).normal_()
        (stop_bit, stop_dist), (message, message_dist), y = receiver(msg, state, None, descriptors)

        check_bernoulli_dist(self, stop_dist)
        check_bernoulli_out(self, stop_bit)
        check_bernoulli_dist(self, message_dist)
        check_bernoulli_out(self, message)
        self.assertEqual(y.size(), (batch_size, self.config.nclasses))

    def test_forward_eval(self):
        batch_size = 3
        receiver = Receiver(self.config)
        receiver.eval()

        msg = torch.FloatTensor(batch_size, self.config.message_in).normal_()
        state = torch.FloatTensor(batch_size, self.config.receiver_hidden_dim).fill_(0)
        descriptors = torch.FloatTensor(self.config.nclasses, self.config.descriptor_dim).normal_()
        (stop_bit, stop_dist), (message, message_dist), y = receiver(msg, state, None, descriptors)

        check_bernoulli_dist(self, stop_dist)
        check_bernoulli_out(self, stop_bit)
        check_bernoulli_dist(self, message_dist)
        check_bernoulli_out(self, message)
        self.assertEqual(y.size(), (batch_size, self.config.nclasses))

    def test_build_pairs(self):
        receiver = Receiver(self.config)

        state = torch.FloatTensor(3, 2)
        state[0] = 0
        state[1] = 1
        state[2] = 2
        descs = torch.FloatTensor(2, 4)
        descs[0] = 5
        descs[1] = 6

        pairs = receiver.build_state_descriptor_pairs(state, descs)

        self.assertEqual(pairs.size(), (6, 6))
        self.assertTrue(pairs[[0, 1], :2].eq(0).all())
        self.assertTrue(pairs[[2, 3], :2].eq(1).all())
        self.assertTrue(pairs[[4, 5], :2].eq(2).all())
        self.assertTrue(pairs[[0, 2, 4], 2:].eq(5).all())
        self.assertTrue(pairs[[1, 3, 5], 2:].eq(6).all())

    def test_reweight_descriptors(self):
        receiver = Receiver(self.config)

        batch_size = 2
        scores = F.softmax(torch.FloatTensor(batch_size, self.config.nclasses).normal_(), dim=1)
        descriptors = torch.FloatTensor(self.config.nclasses, self.config.descriptor_dim).normal_()
        influence = receiver.reweight_descriptors(scores, descriptors)

        self.assertEqual(influence.size(), (batch_size, self.config.descriptor_dim))


if __name__ == '__main__':
    unittest.main()

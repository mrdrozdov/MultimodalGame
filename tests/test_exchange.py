import os
import unittest

import torch
import torch.nn.functional as F

from agents import AgentConfig, Sender, Receiver
from baseline import Baseline
from exchange import Exchange, ExchangeModel


class TestExchange(unittest.TestCase):
    # Convenience Methods

    def exchange_kwargs(self, exchange, batch_size=3, message_size=10):
        message_dist = F.sigmoid(torch.Tensor(batch_size, message_size).normal_())
        message = torch.round(message_dist).detach()
        prediction_log_dist = F.log_softmax(torch.FloatTensor(batch_size, self.config.nclasses).normal_(), dim=1)
        target = prediction_log_dist.argmax(dim=1).unsqueeze(1)
        prediction_log_prob = exchange.loglikelihood(prediction_log_dist, target)
        baseline_scores = prediction_log_dist + prediction_log_dist.clone().normal_()

        return {
            'message': message,
            'message_dist': message_dist,
            'prediction_log_prob': prediction_log_prob,
            'baseline_scores': baseline_scores
        }

    def multistep_exchange_args(self, exchange, batch_size=3, message_size=10, length=3):
        single_kwargs = [self.exchange_kwargs(exchange, batch_size, message_size) for _ in range(length)]
        multi_kwargs = {k: list(map(lambda x: x[k], single_kwargs)) for k in single_kwargs[0].keys()}
        return multi_kwargs

    # Tests

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
        exchange_model = ExchangeModel(self.config)
        baseline_sender = Baseline(self.config, 'sender')
        baseline_receiver = Baseline(self.config, 'receiver')
        exchange = Exchange(exchange_model, sender, receiver, baseline_sender, baseline_receiver, descriptors)

        image = torch.FloatTensor(batch_size, self.config.image_in).normal_()
        results = exchange.exchange(image, length)

        self.assertEqual(len(results), length)

    def test_single_negent(self):
        batch_size = 3
        message_size = 10
        message_dist = F.sigmoid(torch.Tensor(batch_size, message_size).normal_())
        exchange = Exchange(None, None, None, None, None, None)
        negent = exchange.single_entropy_regularization(message_dist)
        
        self.assertEqual(negent.numel(), 1)

    def test_single_exchange_loss(self):
        batch_size = 3
        message_size = 10
        exchange = Exchange(None, None, None, None, None, None)

        loss = exchange.single_exchange_loss(**self.exchange_kwargs(
            exchange, batch_size, message_size))

        self.assertEqual(loss.numel(), 1)

    def test_multi_exchange_loss(self):
        batch_size = 3
        message_size = 10
        length = 3
        regularization = False
        exchange = Exchange(None, None, None, None, None, None)

        def exchange_kwargs():
            return self.exchange_kwargs(exchange, batch_size, message_size)

        multi_kwargs = self.multistep_exchange_args(exchange, batch_size, message_size, length)
        multi_kwargs['entropy_penalty'] = regularization

        loss, regularization_terms = exchange.multistep_exchange_loss_helper(**multi_kwargs)

        self.assertEqual(loss.numel(), 1)
        self.assertEqual(len(regularization_terms), length)
        for negent in regularization_terms:
            self.assertEqual(negent.numel(), 1)


if __name__ == '__main__':
    unittest.main()

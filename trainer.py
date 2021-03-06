import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainerLoss(object):
    pass


class Trainer(object):
    def __init__(self, exchange):
        super(Trainer, self).__init__()
        self.exchange = exchange
        self.length = 10
        self.entropy_penalty = None
        self.adaptive = False

    def extract(self, results, key):
        return [getattr(r, key) for r in results]

    def calculate_loss(self, trainer_loss):
        loss = 0
        loss += trainer_loss.sender_message_loss
        loss += trainer_loss.receiver_message_loss
        loss += trainer_loss.baseline_loss_sender
        loss += trainer_loss.baseline_loss_receiver
        loss += trainer_loss.xent_loss
        if self.adaptive:
            loss += trainer_loss.stop_loss
        return loss

    def run_step(self, image, target):
        # TODO: Add support for adaptive length exchange and for length = 1.
        # TODO: Stop-Bit shouldn't be incorporated for fixed length exchange.
        # TODO: Add regularization term.

        results = self.exchange.exchange(image, self.length)
        if self.adaptive:
            raise NotImplementedError
        else:
            masks = None

        y = self.extract(results, 'y')[-1]
        prediction_log_prob = self.exchange.loglikelihood(y, target.view(target.size(0), 1))

        # Sender Message Loss
        sender_message = self.extract(results, 'sender_message')
        sender_message_dist = self.extract(results, 'sender_message_dist')
        baseline_sender_scores = self.extract(results, 'baseline_sender_scores')
        sender_message_loss, _ = self.exchange.multistep_exchange_loss(sender_message, sender_message_dist, prediction_log_prob, baseline_sender_scores,
            masks, entropy_penalty=self.entropy_penalty)

        # Receiver Message Loss
        # Note: The final receiver message is thrown away.
        receiver_message = self.extract(results[:-1], 'receiver_message')
        receiver_message_dist = self.extract(results[:-1], 'receiver_message_dist')
        baseline_receiver_scores = self.extract(results[:-1], 'baseline_receiver_scores')
        receiver_message_loss, _ = self.exchange.multistep_exchange_loss(receiver_message, receiver_message_dist, prediction_log_prob, baseline_receiver_scores,
            masks, entropy_penalty=self.entropy_penalty)

        # Stop-Bit Loss
        stop_bit = self.extract(results, 'stop_bit')
        stop_dist = self.extract(results, 'stop_dist')
        baseline_receiver_scores = self.extract(results, 'baseline_receiver_scores')
        stop_loss, _ = self.exchange.multistep_exchange_loss(stop_bit, stop_dist, prediction_log_prob, baseline_receiver_scores,
            masks, entropy_penalty=self.entropy_penalty)

        # Baseline Loss
        baseline_loss_sender = self.exchange.multistep_baseline_loss(baseline_sender_scores, prediction_log_prob, masks)
        baseline_loss_receiver = self.exchange.multistep_baseline_loss(baseline_receiver_scores, prediction_log_prob, masks)

        # Cross Entropy Loss
        xent_loss = nn.NLLLoss()(F.log_softmax(y, dim=1), target)

        trainer_loss = TrainerLoss()
        trainer_loss.sender_message_loss = sender_message_loss
        trainer_loss.receiver_message_loss = receiver_message_loss
        trainer_loss.stop_loss = stop_loss
        trainer_loss.baseline_loss_sender = baseline_loss_sender
        trainer_loss.baseline_loss_receiver = baseline_loss_receiver
        trainer_loss.xent_loss = xent_loss
        trainer_loss.y = y

        return trainer_loss


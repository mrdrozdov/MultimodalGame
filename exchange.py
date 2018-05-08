import torch


TINY = 1e-8


class ExchangeResult(object):
    pass


class Exchange(object):
    def __init__(self, sender, receiver, descriptors):
        super(Exchange, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.descriptors = descriptors

    def single_exchange(self, image, message=None, state=None):
        sender_message, sender_message_dist = \
            self.sender(message, None, image, None)
        (stop_bit, stop_dist), (receiver_message, receiver_message_dist), y, new_state = \
            self.receiver(sender_message, state, None, self.descriptors)

        result = ExchangeResult()
        result.sender_message = sender_message
        result.sender_message_dist = sender_message_dist
        result.stop_bit = stop_bit
        result.stop_dist = stop_dist
        result.receiver_message = receiver_message
        result.receiver_message_dist = receiver_message_dist
        result.y = y
        result.new_state = new_state

        return result

    def exchange(self, image, length=10):
        results = []
        for step in range(length):
            if step == 0:
                result = self.single_exchange(image)
                results.append(result)
                continue
            result = self.single_exchange(image, results[-1].receiver_message, results[-1].new_state)
            results.append(result)
        return results

    def loglikelihood(self, log_dist, target):
        """
        Args: 
            log_dist: log softmax scores (N, C) where N is the batch size
              and C is the number of classes
            target: target values (N, 1)
        Output:
            : log likelihood (N)
        """
        return log_dist.gather(1, target)

    def single_entropy_regularization(self, message_dist):
        """Calculate the entropy regularization a single exchange.

        N: Batch size.

        Args:
            message_dist: The bernoulli distribution the message was sampled from. (N, M)
        Output:
            negative_entropy: The regularization term. (N, 1)
        """

        # Must calculate both sides of negative entropy, otherwise it is skewed towards 0.
        initial_negent = (torch.log(message_dist + TINY) * message_dist).sum(1).mean()
        inverse_negent = (torch.log((1. - message_dist) + TINY) * (1. - message_dist)).sum(1).mean()
        negative_entropy = initial_negent + inverse_negent

        return negative_entropy

    def single_exchange_loss(self, message, message_dist, prediction_log_prob, baseline_scores):
        """Calculate the loss component for a single exchange.

        N: Batch size.

        Args:
            message: A binary vector representing the agent's communication or its stop bit. (N, M)
            message_dist: The bernoulli distribution the message was sampled from. (N, M)
            prediction_log_prob: The log likelihood of the correct class. (N, 1)
            baseline_scores: An estimate of the agent's prediction used to reduce variance in REINFORCE. (N, 1)
        Output:
            loss: The loss component being calculated. (N, 1)
        """
        message_log_dist = message * torch.log(message_dist + TINY) + \
            (1 - message) * torch.log(1 - message_dist + TINY)
        message_log_dist = message_log_dist.sum(1)
        weight = prediction_log_prob - baseline_scores.detach()
        if prediction_log_prob.size(0) > 1:
            weight = weight / max(1., torch.std(weight).item())
        loss = torch.mean(-1 * weight * message_log_dist.unsqueeze(1))

        return loss

    def multistep_exchange_loss_masked(self, message, message_dist, prediction_log_prob, baseline_scores, masks, entropy_penalty):
        exchange_length = len(message)
        regularization_terms = [None] * exchange_length

        def mapped_fn(message, message_dist, scores, mask):
            mask_sum = mask.float().sum().item()
            if mask_sum == 0:
                return torch.zeros(1)

            message = message[mask.expand_as(message)]\
                .view(-1, message.size(1))
            message_dist = message_dist[mask.expand_as(message_dist)]\
                .view(-1, message_dist.size(1))
            log_prob = prediction_log_prob[mask.expand_as(prediction_log_prob)]\
                .view(-1, prediction_log_prob.size(1))
            scores = scores[mask.expand_as(scores)]\
                .view(-1, scores.size(1))

            loss = self.single_exchange_loss(message, message_dist, log_prob, scores)

            if entropy_penalty is not None:
                negent = self.single_entropy_regularization(message_dist)
                loss = loss + entropy_penalty * negent

            loss *= mask_sum

            return loss, negent, mask_sum

        output = [mapped_fn(msg, msg_dist, scores, mask)
                  for msg, msg_dist, scores, mask
                  in zip(message, message_dist, baseline_scores, masks)]
        losses = [o[0] for o in output]
        regularization_terms = [o[1] for o in output]
        mask_sums = [o[2] for o in output]
        loss = sum(losses) / sum(mask_sums)

        return loss, regularization_terms

    def multistep_exchange_loss_helper(self, message, message_dist, prediction_log_prob, baseline_scores, entropy_penalty):
        exchange_length = len(message)
        regularization_terms = [None] * exchange_length

        losses = [self.single_exchange_loss(msg, msg_dist, pred_log_prob, baseline_score)
                  for msg, msg_dist, pred_log_prob, baseline_score
                  in zip(message, message_dist, prediction_log_prob, baseline_scores)]

        if entropy_penalty is not None:
            regularization_terms = [self.single_entropy_regularization(msg_dist) for msg_dist in message_dist]
            losses = [l + entropy_penalty * negent for l, negent in zip(losses, regularization_terms)]

        loss = sum(losses) / exchange_length

        return loss, regularization_terms

    def multistep_exchange_loss(self, message, message_dist, prediction_log_prob, baseline_scores, masks, entropy_penalty=None):
        """Calculate the loss for the entire conversation.
        """
        if masks is not None:
            loss, regularization_terms = self.multistep_exchange_loss_masked(
                message, message_dist, prediction_log_prob, baseline_scores, masks, entropy_penalty)
        else:
            loss, regularization_terms = self.multistep_exchange_loss_helper(
                message, message_dist, prediction_log_prob, baseline_scores, entropy_penalty)

        return loss, regularization_terms
        
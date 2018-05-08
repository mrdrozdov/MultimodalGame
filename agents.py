import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class Agent(nn.Module):
    def forward(self, message, state, image, class_descriptors):
        raise NotImplementedError


class AgentConfig(object):
    message_size = 32
    sender_hidden_dim = 256
    receiver_hidden_dim = 256
    receiver_score_dim = 1
    descriptor_dim = 100
    message_in = message_size
    message_out = sender_hidden_dim
    image_in = 512
    image_out = sender_hidden_dim
    binary_in = sender_hidden_dim
    binary_out = message_size
    stop_size = 1
    nclasses = 30


class Sender(Agent):
    def __init__(self, config):
        super(Sender, self).__init__()

        self.config = config

        # Network for communication
        self.message_bias = nn.Parameter(torch.FloatTensor(config.message_out))
        self.message_layer = nn.Linear(config.message_in, config.message_out)
        self.image_layer = nn.Linear(config.image_in, config.image_out)
        self.binary_layer = nn.Linear(config.binary_in, config.binary_out)

    def forward(self, message, state, image, class_descriptors):
        batch_size = image.size(0)

        # Encode image.
        h_img = self.image_layer(image)

        # Encode message.
        if message is not None:
            h_msg = self.message_layer(message)
        else:
            h_msg = self.message_bias.view(1, -1).expand(batch_size, self.config.message_out)

        # Mix information.
        features = self.binary_layer(F.tanh(h_img + h_msg))

        # Specify output distribution.
        dist = F.sigmoid(features)

        # Generate output message.
        if self.training:
            # TODO: Use built-in torch sampling.
            # outp_msg = torch.distributions.Bernoulli(dist).sample().detach()
            probs_ = dist.detach().cpu().numpy()
            outp_msg = torch.from_numpy((np.random.rand(*probs_.shape) < probs_).astype('float32'))
        else:
            outp_msg = torch.round(dist).detach()

        return outp_msg, dist


class Receiver(Agent):
    def __init__(self, config):
        super(Receiver, self).__init__()

        self.config = config

        # RNN Network
        self.rnn = nn.GRUCell(config.message_size, config.receiver_hidden_dim)
        # Network for decisions
        self.stop_layer = nn.Linear(config.receiver_hidden_dim, config.stop_size)
        # Network for class predicitons
        self.y1 = nn.Linear(config.receiver_hidden_dim + config.descriptor_dim, config.receiver_hidden_dim)
        self.y2 = nn.Linear(config.receiver_hidden_dim, config.receiver_score_dim)
        # Network for communication
        self.state_layer = nn.Linear(config.receiver_hidden_dim, config.receiver_hidden_dim)
        self.descriptor_layer = nn.Linear(config.descriptor_dim, config.receiver_hidden_dim, bias=False)
        self.message_layer = nn.Linear(config.receiver_hidden_dim, config.message_size)

    def build_state_descriptor_pairs(self, state, descriptors):
        batch_size, state_size = state.size()
        ndescriptors, descriptor_size = descriptors.size()

        pair_left = state.unsqueeze(2)\
            .expand(batch_size, state_size, ndescriptors)\
            .contiguous()\
            .view(batch_size * ndescriptors, -1)

        pair_right = descriptors.unsqueeze(0)\
            .expand(batch_size, ndescriptors, descriptor_size)\
            .contiguous()\
            .view(batch_size * ndescriptors, -1)

        pair = torch.cat([pair_left, pair_right], dim=1)

        return pair

    def reweight_descriptors(self, scores, descriptors):
        batch_size = scores.size(0)
        ndescriptors, descriptor_size = descriptors.size()

        scores_broadcast = scores.unsqueeze(2)\
            .expand(batch_size, ndescriptors, descriptor_size)
        descriptors_broadcast = descriptors.unsqueeze(0)\
            .expand(batch_size, ndescriptors, descriptor_size)
        reweighted = (scores_broadcast * descriptors_broadcast).sum(1).squeeze(1)

        return reweighted

    def forward(self, message, state, image, class_descriptors):
        batch_size, state_size = state.size()
        ndescriptors, descriptor_size = class_descriptors.size()

        # Encode history.
        new_state = self.rnn(message, state)

        # Predict stop bit.
        stop_dist = F.sigmoid(self.stop_layer(new_state))
        if self.training:
            # TODO: Use built-in torch sampling.
            stop_dist_ = stop_dist.detach().cpu().numpy()
            stop_bit = torch.from_numpy((np.random.rand(*stop_dist_.shape) < stop_dist_).astype('float32'))
        else:
            stop_bit = torch.round(stop_dist).detach()

        # Predict classes.

        ## Expand for pairwise scoring.
        pairs = self.build_state_descriptor_pairs(new_state, class_descriptors)

        ## Predict classes.
        y = self.y1(pairs).clamp(min=0)
        y = self.y2(y).view(batch_size, -1)

        # Obtain communications

        ## Reweight descriptions based on current model confidence.
        y_scores = F.softmax(y, dim=1).detach()
        descriptor_influence = self.reweight_descriptors(y_scores, class_descriptors)

        ## Hidden state for Receiver message.
        message_state = F.tanh(self.state_layer(new_state) + self.descriptor_layer(descriptor_influence))

        ## Message distribution.
        message_dist = F.sigmoid(self.message_layer(message_state))

        ## Generate output message.
        if self.training:
            # TODO: Use built-in torch sampling.
            message_dist_ = message_dist.detach().cpu().numpy()
            outp_msg = torch.from_numpy((np.random.rand(*message_dist_.shape) < message_dist_).astype('float32'))
        else:
            outp_msg = torch.round(message_dist).detach()

        return (stop_bit, stop_dist), (outp_msg, message_dist), y

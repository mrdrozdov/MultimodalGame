import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class Agent(nn.Module):
    def forward(self, message, state, image, class_descriptors):
        raise NotImplementedError


class AgentConfig(object):
    message_in = 32
    message_out = 256
    image_in = 512
    image_out = 256
    binary_in = 256
    binary_out = 32


class Sender(Agent):
    def __init__(self, config):
        super(Sender, self).__init__()

        self.config = config

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

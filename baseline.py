import torch
import torch.nn as nn


class Baseline(nn.Module):
    """Baseline
    """

    def __init__(self, config, target):
        super(Baseline, self).__init__()
        self.config = config
        self.image_size = config.image_in
        self.message_size = config.message_size
        self.receiver_hidden_dim = config.receiver_hidden_dim
        self.baseline_dim = config.baseline_dim

        if target == 'sender':
            nfeatures = self.image_size + self.message_size
        elif target == 'receiver':
            nfeatures = self.message_size + self.receiver_hidden_dim
        else:
            raise ValueError("Incompatible target '{}'.".format(target))

        # Additional layers on top of feature extractor
        self.linear1 = nn.Linear(nfeatures, self.baseline_dim)
        self.linear2 = nn.Linear(self.baseline_dim, 1)

    def forward(self, image, message, state):
        """Estimate agent's loss based on the agent's input.

        Args:
            image: Image features.
            message: Communication message.
            state: Hidden state (used when agent is the Receiver).
        Output:
            score: An estimate of the agent's loss.
        """
        features = []
        if image is not None:
            features.append(image)
        if message is not None:
            features.append(message)
        if state is not None:
            features.append(state)
        features = torch.cat(features, 1)
        hidden = self.linear1(features).clamp(min=0)
        pred_score = self.linear2(hidden)
        return pred_score

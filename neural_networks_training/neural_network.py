from torch import nn
import torch.nn.functional as F

class FFN(nn.Module):
    """
    A simple Feedforward Neural Network with two hidden layers.
    """
    def __init__(self, n_features, n_hl, n_outputs):
        super(FFN, self).__init__()
        # Define hidden layers separately
        self.hidden_layers = nn.Sequential(
            nn.Linear(n_features, n_hl),
            nn.ReLU(),
            nn.Linear(n_hl, n_hl),
            nn.ReLU()
        )
        # Define the output layer separately
        self.output_layer = nn.Linear(n_hl, n_outputs)

    def forward(self, x):
        # Pass input through hidden layers
        x = self.hidden_layers(x)
        # Get logits from the output layer
        logits = self.output_layer(x)
        # Apply the required truncated identity function
        return F.hardtanh(logits, min_val=0, max_val=1)
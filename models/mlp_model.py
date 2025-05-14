import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes):
        super(MLPModel, self).__init__()
        flattened_dim = input_dim * seq_len  # e.g., 52 * 20 = 1040
        self.model = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (batch, seq_len, input_dim) → (batch, seq_len * input_dim)
        return self.model(x)

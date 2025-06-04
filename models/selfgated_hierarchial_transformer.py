import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Equation (2), (3): sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # P_{t,2i}
        pe[:, 1::2] = torch.cos(position * div_term)  # P_{t,2i+1}
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # Equation (4): H_enc^(0) = H^(0) + P
        return x + self.pe[:, :x.size(1)].to(x.device)


class SelfGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Equation (10): G = sigmoid(H_pool W_g + b_g)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Equation (11): H_gated = H_pool ⊙ G
        return x * self.gate(x)


class SelfGatedHierarchicalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4,
                 num_layers_low=3, num_layers_high=3,
                 dim_feedforward=128, dropout=0.001,
                 pool_output_size=10, num_classes=21):
        super().__init__()

        # Equation (1): Linear projection X W_proj + b_proj
        self.input_proj = nn.Linear(input_dim, d_model)

        # Equation (4): positional encoding added after projection
        self.pos_encoder = PositionalEncoding(d_model)

        # Equation (5–9): local-level transformer encoder (L_l layers)
        encoder_layer_low = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder_low = nn.TransformerEncoder(encoder_layer_low, num_layers=num_layers_low)

        # Equation (9): AdaptiveAvgPool1d(H_l^(L_l)^T)^T → compress time dimension
        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        # Equation (10–11): self-gating applied to pooled output
        self.self_gate = SelfGating(d_model)

        # Equation (12): high-level transformer encoder (L_h layers)
        encoder_layer_high = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder_high = nn.TransformerEncoder(encoder_layer_high, num_layers=num_layers_high)

        # Equation (14): classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),           # W_c1
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)        # W_c2
        )

    def forward(self, x):
        B, T, F = x.shape

        # Project input to model space: Equation (1)
        x = self.input_proj(x)

        # Add positional encoding: Equation (4)
        x = self.pos_encoder(x)

        # Local transformer encoding: Equation (5–9)
        low_out = self.encoder_low(x)

        # Temporal pooling: Equation (9)
        pooled = self.pool(low_out.transpose(1, 2)).transpose(1, 2)

        # Self-gating: Equation (10–11)
        gated = self.self_gate(pooled)

        # High-level transformer: Equation (12)
        high_out = self.encoder_high(gated)

        # Per-timestep logits: intermediate output before mean pooling
        logits_per_timestep = self.classifier(high_out)

        # Temporal mean pooling: Equation (13)
        final_logits = logits_per_timestep.mean(dim=1)

        return final_logits  # Equation (14): output logits ∈ ℝ^{B × num_classes}

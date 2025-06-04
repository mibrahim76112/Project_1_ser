import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Eq. (2) and (3): sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Eq. (2): P_{t,2i}
        pe[:, 1::2] = torch.cos(position * div_term)  # Eq. (3): P_{t,2i+1}
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # Eq. (4): Add positional encoding: H_enc^(0) = H^(0) + P
        return x + self.pe[:, :x.size(1)].to(x.device)


class SelfGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Eq. (11): G = σ(H_pool W_g + b_g)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Eq. (12): H_gated = H_pool ⊙ G
        return x * self.gate(x)


class SelfGatedHierarchicalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4,
                 num_layers_low=3, num_layers_high=3,
                 dim_feedforward=128, dropout=0.001,
                 pool_output_size=10, num_classes=21):
        super().__init__()

        # Eq. (1): Linear projection X W_proj + b_proj
        self.input_proj = nn.Linear(input_dim, d_model)

        # Eq. (4): Add positional encoding after projection
        self.pos_encoder = PositionalEncoding(d_model)

        # Eq. (5–9): Local Transformer encoder block (L_ell layers)
        encoder_layer_low = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder_low = nn.TransformerEncoder(encoder_layer_low, num_layers=num_layers_low)

        # Eq. (10): Adaptive average pooling to reduce time dimension
        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        # Eq. (11)–(12): Self-gating after pooling
        self.self_gate = SelfGating(d_model)

        # Eq. (13): High-level Transformer encoder block (L_h layers)
        encoder_layer_high = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder_high = nn.TransformerEncoder(encoder_layer_high, num_layers=num_layers_high)

        # Eq. (15)–(17): Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),  # Eq. (15): z W_c1 + b_c1
            nn.ReLU(),
            nn.Dropout(0.2),          # Eq. (16): Dropout
            nn.Linear(128, num_classes)  # Eq. (17): h_drop W_c2 + b_c2
        )

    def forward(self, x):
        B, T, F = x.shape

        # Eq. (1): Linear projection
        x = self.input_proj(x)

        # Eq. (4): Add positional encoding
        x = self.pos_encoder(x)

        # Eq. (5–9): Local Transformer encoding
        low_out = self.encoder_low(x)

        # Eq. (10): Adaptive pooling: H_pool = AdaptiveAvgPool1d(H_ell^T)^T
        pooled = self.pool(low_out.transpose(1, 2)).transpose(1, 2)

        # Eq. (11)–(12): Self-gating
        gated = self.self_gate(pooled)

        # Eq. (13): High-level Transformer encoding
        high_out = self.encoder_high(gated)

        # Eq. (15)–(17): Classifier on each timestep
        logits_per_timestep = self.classifier(high_out)

        # Eq. (14): Mean over time: z = (1/T') ∑ H_h[:, t, :]
        final_logits = logits_per_timestep.mean(dim=1)

        return final_logits  # Final prediction: ŷ ∈ ℝ^{B × num_classes}

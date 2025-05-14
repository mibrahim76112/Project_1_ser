import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # not a parameter, but moves with device

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=2, ff_dim=128, dropout=0.1, max_len=500):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len)

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, features]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, features]
        x = self.pos_encoder(x)  # Add positional information

        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)

        x = x.permute(1, 2, 0)  # [batch_size, features, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch_size, features]
        x = self.fc(x)  # [batch_size, num_classes]
        return x
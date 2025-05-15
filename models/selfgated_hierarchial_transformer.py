import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class SelfGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate_val = self.gate(x)  
        gated_x = x * gate_val   
        return gated_x

class SelfGatedHierarchicalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4,
                 num_layers_low=2, num_layers_high=2,
                 dim_feedforward=128, dropout=0.1,
                 pool_output_size=10, num_classes=21):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer_low = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder_low = nn.TransformerEncoder(encoder_layer_low, num_layers=num_layers_low)

        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        self.self_gate = SelfGating(d_model)  

        encoder_layer_high = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder_high = nn.TransformerEncoder(encoder_layer_high, num_layers=num_layers_high)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
      
        B, T, F = x.shape
        x = self.input_proj(x)  
        x = self.pos_encoder(x)

        low_out = self.encoder_low(x)  

      
        pooled = self.pool(low_out.transpose(1, 2)).transpose(1, 2)  

        # Apply self-gating
        gated = self.self_gate(pooled)  

        high_out = self.encoder_high(gated)  

        final_repr = high_out.mean(dim=1) 

        logits = self.classifier(final_repr)
        return logits

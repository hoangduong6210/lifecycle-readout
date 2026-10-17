"""Inductive temporal mixer over strictly past event features."""
import torch
from torch import nn

class BochnerTimeEncoder(nn.Module):
    def __init__(self, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        freq = 1.0 / (10.0 ** (torch.arange(0, time_dim, 2).float() / time_dim * 6.0))
        self.register_buffer("freq", freq)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        args = dt.unsqueeze(-1) * self.freq.unsqueeze(0).unsqueeze(0)
        sin_enc = torch.sin(args)
        cos_enc = torch.cos(args)
        return torch.cat([cos_enc, sin_enc], dim=-1)


class BaseTemporalPredictor(nn.Module):
    """Inductive Temporal Graph Predictor with Bochner Time-Encoded MLP-Mixer."""

    def __init__(self, num_nodes: int = 0, feat_dim: int = 172, hidden_dim: int = 128, time_dim: int = 32, max_neighbors: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_encoder = BochnerTimeEncoder(time_dim)

        in_dim = feat_dim + time_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(max_neighbors, max_neighbors),
            nn.GELU(),
            nn.Linear(max_neighbors, max_neighbors),
        )

        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.empty_node = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.empty_node, std=0.02)

        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_node(self, neighbor_feats: torch.Tensor, neighbor_dt: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        time_feats = self.time_encoder(neighbor_dt)
        in_feats = torch.cat([neighbor_feats, time_feats], dim=-1)
        h = self.in_proj(in_feats)

        mask_f = neighbor_mask.unsqueeze(-1).float()
        h = h * mask_f

        h_norm = self.token_norm(h).transpose(1, 2)
        h_token = self.token_mlp(h_norm).transpose(1, 2)
        h = h + h_token * mask_f

        h_norm = self.channel_norm(h)
        h_chan = self.channel_mlp(h_norm)
        h = h + h_chan * mask_f

        denom = mask_f.sum(dim=1).clamp(min=1.0)
        h_pool = (h * mask_f).sum(dim=1) / denom

        is_empty = (mask_f.sum(dim=1) == 0).float()
        h_pool = h_pool + is_empty * self.empty_node
        return h_pool

    def forward(self, u_feats, u_dt, u_mask, v_feats, v_dt, v_mask):
        h_u = self.encode_node(u_feats, u_dt, u_mask)
        h_v = self.encode_node(v_feats, v_dt, v_mask)
        pair_cat = torch.cat([h_u, h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
        h_uv = self.pair_proj(pair_cat)
        base_logits = self.scorer(h_uv).squeeze(-1)
        return h_uv, base_logits

"""
SR-GNN v2 — Improved based on experiment diagnostics

Root cause analysis of v1:
  1. CSN over-filters → residual gating (feat always passes through, gate modulates)
  2. DRGC lacks bidirectional coupling → add symmetric msg with separate dst→src path
  3. TIP compresses ALL nodes blindly → selective: only compress low-resonance nodes
  4. NSCP only penalises DEATH → add full state-aware scoring + edge state embedding
  5. Message passing too shallow → add a second message aggregation round per event

Key changes:
  CSN  → ResidualCSN:     gate = α·filtered + (1-α)·raw,  α learned per-event
  ECTG → ECTG unchanged   (already strongest module)
  DRGC → DRGC_v2:         bidirectional coupled GRU + time-decay on stale memories
  TIP  → TIP_v2:          selective compression (only low-intensity nodes) + contrastive
  NSCP → NSCP_v2:         edge-state-aware predictor + state consistency loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

from .sr_gnn import (
    IDLE, BIRTH, REINFORCE, DECAY, DEATH,
    VALID_TRANSITIONS,
    TimeEncoder,
    NodeMemoryStore,
    EdgeStateStore,
    ECTG,
)


# ────────────────────────────────────────────────────────────
# Layer 1 v2 — ResidualCSN
# ────────────────────────────────────────────────────────────

class ResidualCSN(nn.Module):
    """
    Fix: v1 CSN multiplied feat × salience → zeros out useful info.
    v2:  output = feat + salience · transform(feat)
    Salience now *enhances* novel events rather than *suppressing* common ones.
    """
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.feat_dim = feat_dim
        if feat_dim > 0:
            self.feat_transform = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, feat_dim)
            )
            self.feat_proj = nn.Linear(feat_dim, hidden)
            self.register_buffer("feat_ema", torch.zeros(hidden))
        else:
            self.feat_transform = None
            self.feat_proj = None

        # Temporal novelty
        self.time_gate = nn.Sequential(
            nn.Linear(1, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1), nn.Sigmoid()
        )
        # Combined salience
        self.alpha_net = nn.Sequential(
            nn.Linear(2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1), nn.Sigmoid()
        )

    def forward(self, feat: Tensor, delta_t: Tensor) -> Tuple[Tensor, Tensor]:
        B = feat.size(0)
        device = feat.device

        # Temporal novelty
        log_dt = torch.log1p(delta_t.float()).unsqueeze(-1)
        nov_t = self.time_gate(log_dt)                              # (B,1)

        # Semantic novelty
        if self.feat_proj is not None and feat.size(-1) > 0:
            proj = F.normalize(self.feat_proj(feat), dim=-1)
            if self.training:
                self.feat_ema = self.feat_ema * 0.99 + proj.detach().mean(0) * 0.01
            cos_sim = (proj * F.normalize(self.feat_ema.unsqueeze(0), dim=-1)
                       ).sum(-1, keepdim=True).clamp(-1, 1)
            nov_s = (1.0 - (cos_sim + 1.0) / 2.0)                  # (B,1)
        else:
            nov_s = torch.ones(B, 1, device=device)

        # Salience = mixing ratio
        alpha = self.alpha_net(torch.cat([nov_t, nov_s], -1)).squeeze(-1)  # (B,)

        # RESIDUAL: raw feat always passes through, enhanced by salience
        if self.feat_transform is not None and feat.size(-1) > 0:
            enhanced = self.feat_transform(feat)                     # (B, feat_dim)
            gated = feat + alpha.unsqueeze(-1) * enhanced            # residual add
        else:
            gated = feat

        return gated, alpha


# ────────────────────────────────────────────────────────────
# Layer 3 v2 — TIP_v2
# ────────────────────────────────────────────────────────────

class TIP_v2(nn.Module):
    """
    Fix: v1 TIP compressed ALL nodes uniformly → near-zero impact.
    v2:  Selective compression — only low-resonance nodes get pushed toward prior.
         High-resonance nodes pass through untouched.
         + Temporal decay: stale memories decay toward zero.
    """
    def __init__(self, hidden: int, beta: float = 0.001):
        super().__init__()
        self.beta = beta
        self.mu_net = nn.Linear(hidden, hidden)
        self.lv_net = nn.Linear(hidden, hidden)

        # Resonance gate: determines how much to compress each node
        self.compress_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )
        # Edge-level resonance gate
        self.res_gate = nn.Sequential(
            nn.Linear(1, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1), nn.Sigmoid()
        )

    def forward(self, h: Tensor, staleness: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        """
        h: (N, hidden)
        staleness: (N,)  time since last update — stale nodes compress more
        Returns: h_parsed (N, hidden), kl_loss scalar
        """
        mu = self.mu_net(h)
        lv = self.lv_net(h).clamp(-4, 4)

        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        else:
            z = mu

        # Per-node compression strength: low-info nodes compress more
        compress_strength = self.compress_gate(h.detach())           # (N, 1)

        # Selective: blend compressed z with original h
        # High-resonance nodes → keep h; low-resonance → use z
        h_parsed = compress_strength * z + (1.0 - compress_strength) * h

        # Temporal decay: stale nodes decay toward zero
        if staleness is not None:
            decay = torch.exp(-0.01 * staleness.float()).unsqueeze(-1)  # (N, 1)
            h_parsed = h_parsed * decay

        # KL only on compressed portion (weighted by compress_strength)
        kl_per_node = -0.5 * (1 + lv - mu.pow(2) - lv.exp())        # (N, H)
        kl = (compress_strength * kl_per_node).mean() * self.beta

        return h_parsed, kl

    def resonance_mask(self, intensity: Tensor) -> Tensor:
        return self.res_gate(intensity.unsqueeze(-1)).squeeze(-1)


# ────────────────────────────────────────────────────────────
# Layer 3 v2 — DRGC_v2
# ────────────────────────────────────────────────────────────

class DRGC_v2(nn.Module):
    """
    Fix: v1 used same GRU for src and dst → poor coupling.
    v2:  Separate src/dst message networks + mutual update (like JODIE).
         + Time-conditioned memory decay before update.
    """
    def __init__(self, feat_dim: int, hidden: int, beta: float = 0.001):
        super().__init__()
        self.hidden = hidden
        self.time_enc = TimeEncoder(hidden)

        # Separate message functions for src→dst and dst→src
        msg_in = hidden * 2 + hidden + hidden + feat_dim  # h_self, h_other, edge_ctx, t_enc, feat
        self.msg_s2d = nn.Sequential(
            nn.Linear(msg_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.msg_d2s = nn.Sequential(
            nn.Linear(msg_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        # Separate GRU updaters (like JODIE's coupled RNN)
        self.gru_src = nn.GRUCell(hidden, hidden)
        self.gru_dst = nn.GRUCell(hidden, hidden)

        # Memory time-decay: project time gap into decay factor
        self.decay_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.Sigmoid()
        )

        self.tip = TIP_v2(hidden, beta)

    def forward(self, h_src: Tensor, h_dst: Tensor,
                feat: Tensor, edge_ctx: Tensor,
                dt_src: Tensor, dt_dst: Tensor, intensity: Tensor,
                compress_nodes: Tensor, compress_staleness: Optional[Tensor]
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns: new_h_src, new_h_dst, parsed_h, tip_loss
        """
        # Time encoding
        t_enc_s = self.time_enc(dt_src)
        t_enc_d = self.time_enc(dt_dst)

        # Time decay: stale memories shrink before update
        decay_s = self.decay_net(t_enc_s)                            # (B, H)
        decay_d = self.decay_net(t_enc_d)
        h_src_d = h_src * decay_s
        h_dst_d = h_dst * decay_d

        # Resonance mask on messages
        r_mask = self.tip.resonance_mask(intensity).unsqueeze(-1)    # (B, 1)

        # Bidirectional messages
        msg_s = self.msg_s2d(torch.cat([h_src_d, h_dst_d, edge_ctx, t_enc_s, feat], -1))
        msg_d = self.msg_d2s(torch.cat([h_dst_d, h_src_d, edge_ctx, t_enc_d, feat], -1))
        msg_s = msg_s * r_mask
        msg_d = msg_d * r_mask

        # Coupled GRU update
        new_h_src = self.gru_src(msg_d, h_src)  # dst's message updates src
        new_h_dst = self.gru_dst(msg_s, h_dst)  # src's message updates dst

        # TIP selective compression
        parsed_h, kl = self.tip(compress_nodes, compress_staleness)

        return new_h_src, new_h_dst, parsed_h, kl


# ────────────────────────────────────────────────────────────
# Layer 4 v2 — NSCP_v2
# ────────────────────────────────────────────────────────────

class NSCP_v2(nn.Module):
    """
    Fix: v1 only penalised DEATH state → weak causal signal.
    v2:  Edge state embedding fed into predictor.
         + State consistency loss: predicted state should match ECTG state.
         + CCS uses full state distribution, not just death.
    """
    def __init__(self, hidden: int):
        super().__init__()
        # Edge state → embedding
        self.state_embed = nn.Sequential(
            nn.Linear(8, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4)
        )
        # Predictor uses src_emb, dst_emb, state_emb
        pred_in = hidden * 2 + hidden // 4
        self.predictor = nn.Sequential(
            nn.Linear(pred_in, hidden), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1)
        )
        # State consistency: predict what state the edge should be in
        # self.state_predictor = nn.Linear(hidden * 2, 5) # REMOVED: will be external StateOracle

    def forward(self, src_emb: Tensor, dst_emb: Tensor,
                neg_dst_emb: Tensor, edge_state: Tensor,
                state_logits: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        st_emb = self.state_embed(edge_state)                       # (B, H//4)

        # Use SAME state embedding for both pos and neg to avoid feature leakage.
        # The predictor must learn from node embeddings, not from state presence.
        pos_in = torch.cat([src_emb, dst_emb, st_emb], -1)
        pos_score = self.predictor(pos_in).squeeze(-1)

        neg_in = torch.cat([src_emb, neg_dst_emb, st_emb], -1)    # same st_emb
        neg_score = self.predictor(neg_in).squeeze(-1)

        # Causal constraints:
        # 1. DEATH/DECAY edges should have lower scores
        decay_death = edge_state[:, DECAY] + edge_state[:, DEATH]
        # 2. REINFORCE edges should have higher scores
        reinforce = edge_state[:, REINFORCE]

        causal_loss = (
            (decay_death * F.relu(pos_score)).mean()                 # penalise predicting dying edges
            + (reinforce * F.relu(-pos_score)).mean() * 0.5          # reward predicting strong edges
        )

        # State consistency loss: node embeddings (via StateOracle) should predict edge state
        # This loss trains the StateOracle
        state_target = edge_state[:, :5].detach()                    # soft target
        state_loss = F.kl_div(
            F.log_softmax(state_logits, -1), state_target, reduction='batchmean'
        ) * 0.1

        total_causal = causal_loss + state_loss

        # CCS: 1 - (probability of causally-invalid prediction)
        ccs = 1.0 - (decay_death * torch.sigmoid(pos_score).detach()).clamp(0, 1)

        return pos_score, neg_score, total_causal, ccs


# ────────────────────────────────────────────────────────────
# Full SR-GNN v2
# ────────────────────────────────────────────────────────────

class SRGNN_v2(nn.Module):
    def __init__(self, num_nodes: int, feat_dim: int, hidden: int = 128,
                 tip_beta: float = 0.001,
                 lambda_tip: float = 0.01, lambda_causal: float = 0.1, lambda_trans: float = 0.05,
                 lambda_distill: float = 0.2, # New hyperparam for hybrid loss
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_nodes      = num_nodes
        self.feat_dim       = feat_dim
        self.hidden         = hidden
        self.lambda_tip     = lambda_tip
        self.lambda_causal  = lambda_causal
        self.lambda_trans   = lambda_trans
        self.lambda_distill = lambda_distill
        self.device         = device
        self._feat_in       = max(feat_dim, 1)

        self.csn  = ResidualCSN(self._feat_in, hidden)
        self.ectg = ECTG(self._feat_in, hidden)
        self.drgc = DRGC_v2(self._feat_in, hidden, tip_beta)
        self.nscp = NSCP_v2(hidden)

        self.node_mem  = NodeMemoryStore(num_nodes, hidden, device)
        self.edge_mem  = EdgeStateStore(num_nodes, hidden, device)

        # StateOracle: The "Teacher" model for state distillation
        self.state_oracle = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5)
        ).to(device)

    def reset(self):
        self.node_mem.reset()
        self.edge_mem.reset()

    def forward(self, src: Tensor, dst: Tensor, t: Tensor,
                feat: Tensor, neg_dst: Tensor,
                rel_type: Optional[Tensor] = None) -> Dict[str, Tensor]:

        device = self.device
        B      = src.size(0)

        if feat.shape[-1] == 0:
            feat = torch.zeros(B, 1, device=device)
        elif feat.shape[-1] < self._feat_in:
            feat = F.pad(feat, (0, self._feat_in - feat.shape[-1]))

        # ── L1: ResidualCSN ──
        dt_src = self.node_mem.delta_t(src, t)
        feat_g, sal = self.csn(feat, dt_src)

        # ── L2: ECTG ──
        # Returns its own logits ("student_logits") and the heuristic target
        edge_st = self.edge_mem.get_batch(src, dst)
        new_est, edge_ctx, student_logits, heuristic_target_dist = self.ectg(
            feat_g, sal, dt_src, edge_st)
        self.edge_mem.update_batch(src, dst, new_est)
        intensity = new_est[:, 5]

        # ── L3: DRGC_v2 + TIP_v2 ──
        h_src = self.node_mem.get(src)
        h_dst = self.node_mem.get(dst)
        dt_dst = self.node_mem.delta_t(dst, t)

        all_idx = torch.unique(torch.cat([src, dst]))
        all_h = self.node_mem.get(all_idx)
        all_staleness = (t.max().float() - self.node_mem.last_t[all_idx]).clamp(0)

        new_h_src, new_h_dst, parsed_h, kl = self.drgc(
            h_src, h_dst, feat_g, edge_ctx, dt_src, dt_dst,
            intensity, all_h, all_staleness
        )

        # === BƯỚC MỚI: "THẦY" STATEORACLE ĐƯA RA LỜI TIÊN TRI ===
        teacher_logits = self.state_oracle(torch.cat([new_h_src, new_h_dst], dim=-1))

        # ── L4: NSCP_v2 — score BEFORE memory update (prevent leakage) ──
        # Pass teacher_logits to NSCP to train the StateOracle
        neg_emb = self.node_mem.get(neg_dst)   # neg has no event → use current memory
        pos_sc, neg_sc, c_loss, ccs = self.nscp(
            new_h_src, new_h_dst, neg_emb, new_est, teacher_logits
        )

        # NOW update memory (after scoring)
        self.node_mem.set(all_idx, parsed_h)
        self.node_mem.set(src, new_h_src)
        self.node_mem.set(dst, new_h_dst)
        self.node_mem.update_time(torch.cat([src, dst]), torch.cat([t, t]))

        # ── Loss ──
        pred_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_sc, neg_sc]),
            torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])
        )

        # HYBRID transition loss
        s_teacher = F.softmax(teacher_logits, dim=-1).detach()
        log_s_student = F.log_softmax(student_logits, dim=-1)

        loss_heuristic = F.kl_div(log_s_student, heuristic_target_dist, reduction='batchmean')
        loss_predictive = F.kl_div(log_s_student, s_teacher, reduction='batchmean')

        trans_loss = (1 - self.lambda_distill) * loss_heuristic + self.lambda_distill * loss_predictive

        total = (pred_loss
                 + self.lambda_tip * kl
                 + self.lambda_causal * c_loss
                 + self.lambda_trans * trans_loss)

        return {
            "pos_score": pos_sc, "neg_score": neg_sc,
            "loss": total, "pred_loss": pred_loss,
            "tip_loss": kl, "causal_loss": c_loss,
            "trans_loss": trans_loss,
            "ccs": ccs.mean(), "salience": sal.mean(),
        }

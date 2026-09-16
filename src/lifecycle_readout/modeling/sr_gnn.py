"""
SR-GNN (Resonance-Symbolic Graph Neural Network) — optimised version

Pipeline:
  [L1] CSN  — Cognitive Signal Normalization        (semantic filtering)
  [L2] ECTG — Event-Causal Temporal Graph           (causal state machine)
  [L3] DRGC — Dual-Reasoning Graph Core  + TIP      (message passing + parsimony)
  [L4] NSCP — Neuro-Symbolic Causal Policy           (transition masking)

Optimisations for M1 / CPU:
  - Vectorised edge-state tensor (no Python dict loops)
  - All ops on single device (CPU or MPS selected at build time)
  - Batch-level parallelism; GRU cell → matmul-friendly
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

# State indices
IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4

# Causal transition mask  [from_state, to_state] — stays on CPU, indexed via .cpu()
#
# TRIDIAGONAL / BI-DIRECTIONAL adjacency: each state reaches {previous, self, next}.
# Corrected 2026-05-31 (design correction) from the prior FORWARD-ONLY chain, which
# allowed only {self, next} and thus zeroed gradient toward any state below the
# current rung (-1e9 hard mask) — the structural pin behind the BIRTH/REINFORCE
# frontier attractor. The tridiagonal form ADDS the backward (recovery/dormancy)
# rungs while STILL preserving causal admissibility:
#   - IDLE has no "previous" → only {IDLE, BIRTH}; cannot skip to REINFORCE/DECAY/DEATH.
#   - No 2-step jumps: an edge still cannot be born-and-die in one step (BIRTH↛DEATH),
#     nor die-before-born (IDLE↛DEATH). Single-rung moves only.
#   - Backward rungs now ADMISSIBLE: BIRTH→IDLE (dormancy), REINFORCE→BIRTH,
#     DECAY→REINFORCE (recovery), DEATH→DECAY. These supply the bidirectional flow
#     that lets the chained argmax (and its adjacent-state gradient) leave the
#     frontier instead of monotonically climbing to BIRTH.
#   - DEATH is no longer terminal: DEATH→DECAY is admissible (single-rung), matching
#     the intended bi-directional FSM (revival flows DEATH→DECAY→REINFORCE→…).
VALID_TRANSITIONS = torch.tensor([
    [True,  True,  False, False, False],  # IDLE       : no prev → {self, BIRTH}
    [True,  True,  True,  False, False],  # BIRTH      : {IDLE, self, REINFORCE}
    [False, True,  True,  True,  False],  # REINFORCE  : {BIRTH, self, DECAY}
    [False, False, True,  True,  True ],  # DECAY      : {REINFORCE, self, DEATH}
    [False, False, False, True,  True ],  # DEATH      : {DECAY, self}
], dtype=torch.bool)


# ────────────────────────────────────────────────────────────
# Utility modules
# ────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(1, dim)
        nn.init.normal_(self.w.weight, std=0.1)

    def forward(self, t: Tensor) -> Tensor:          # (B,) → (B, dim)
        return torch.cos(self.w(t.unsqueeze(-1).float()))


# ────────────────────────────────────────────────────────────
# Vectorised state stores  (tensor-based, no Python loops)
# ────────────────────────────────────────────────────────────

class NodeMemoryStore:
    """Dense node memory [num_nodes, hidden] + last-seen timestamp."""
    def __init__(self, num_nodes: int, hidden: int, device: torch.device):
        self.device = device
        self.memory = torch.zeros(num_nodes, hidden, device=device)
        self.last_t  = torch.zeros(num_nodes,        device=device)

    def get(self, idx: Tensor) -> Tensor:
        return self.memory[idx]

    def set(self, idx: Tensor, h: Tensor):
        self.memory[idx] = h.detach()

    def delta_t(self, idx: Tensor, t: Tensor) -> Tensor:
        return (t.float() - self.last_t[idx]).clamp(min=0.0)

    def update_time(self, idx: Tensor, t: Tensor):
        self.last_t[idx] = t.float()

    def reset(self):
        self.memory.zero_()
        self.last_t.zero_()


class EdgeStateStore:
    """
    Sparse edge state store.
    Key: linearised (u*N + v) — allocated lazily, O(1) lookup via dict of tensors.
    State vector per edge: [state_logits(5) | intensity(1) | volatility(1) | lifecycle(1)] = 8 dims
    """
    def __init__(self, num_nodes: int, hidden: int, device: torch.device):
        self.N      = num_nodes
        self.device = device
        # edge_key (int64 scalar) → index in state_table
        self._key_to_idx: Dict[int, int] = {}
        self._state_table: list = []          # will be converted to tensor on first use
        self._dirty = False

    def _make_init_state(self) -> Tensor:
        s = torch.zeros(8, device=self.device)
        s[IDLE] = 1.0
        return s

    def get_batch(self, src: Tensor, dst: Tensor) -> Tensor:
        """Return (B, 8) state vectors for a batch of edges."""
        B = src.size(0)
        states = torch.zeros(B, 8, device=self.device)
        src_c = src.tolist()
        dst_c = dst.tolist()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            if key not in self._key_to_idx:
                idx = len(self._state_table)
                self._key_to_idx[key] = idx
                self._state_table.append(self._make_init_state())
            states[i] = self._state_table[self._key_to_idx[key]]
        return states

    def update_batch(self, src: Tensor, dst: Tensor, new_states: Tensor):
        """Write (B, 8) new states back."""
        src_c = src.tolist()
        dst_c = dst.tolist()
        ns = new_states.detach()
        for i, (u, v) in enumerate(zip(src_c, dst_c)):
            key = u * self.N + v
            if key in self._key_to_idx:
                self._state_table[self._key_to_idx[key]] = ns[i]

    def reset(self):
        self._key_to_idx.clear()
        self._state_table.clear()


# ────────────────────────────────────────────────────────────
# Layer 1 — CSN
# ────────────────────────────────────────────────────────────

class CSN(nn.Module):
    """Golden Semantic Entropy++ filter — soft salience gate per event."""
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.feat_dim = feat_dim

        # novelty branch: log(Δt+1) → scalar
        self.novelty_fc = nn.Linear(1, 1)

        # feature branch: project & compare to running mean
        if feat_dim > 0:
            self.feat_proj = nn.Linear(feat_dim, hidden)
            self.register_buffer("feat_mean", torch.zeros(hidden))
            self._count = 0
        else:
            self.feat_proj = None

        # combine 2 scores → gate
        self.gate = nn.Sequential(
            nn.Linear(2, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

    def forward(self, feat: Tensor, delta_t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        feat     : (B, feat_dim)
        delta_t  : (B,)
        Returns  : gated_feat (B, feat_dim),  salience (B,)
        """
        # 1. Temporal novelty
        log_dt = torch.log1p(delta_t.float()).unsqueeze(-1)       # (B,1)
        nov_t  = torch.sigmoid(self.novelty_fc(log_dt))           # (B,1)

        # 2. Semantic novelty (cosine distance from running mean)
        if self.feat_proj is not None and feat.size(-1) > 0:
            proj = F.normalize(self.feat_proj(feat), dim=-1)      # (B, H)
            # update running mean (detached)
            if self.training:
                self.feat_mean = (self.feat_mean * 0.99
                                  + proj.detach().mean(0) * 0.01)
            sim = (proj * F.normalize(self.feat_mean.unsqueeze(0), dim=-1)
                   ).sum(-1, keepdim=True).clamp(-1, 1)           # (B,1)
            nov_s = (1.0 - (sim + 1) / 2)                         # (B,1)  in [0,1]
        else:
            nov_s = torch.ones(feat.size(0), 1, device=feat.device)

        # 3. Gate
        scores   = torch.cat([nov_t, nov_s], dim=-1)              # (B,2)
        salience = self.gate(scores).squeeze(-1)                   # (B,)

        gated = feat * salience.unsqueeze(-1) if feat.size(-1) > 0 else feat
        return gated, salience


# ────────────────────────────────────────────────────────────
# Layer 2 — ECTG
# ────────────────────────────────────────────────────────────

class ECTG(nn.Module):
    """Event-Causal Temporal Graph: updates edge states via causal state machine.

    Transition supervision: edges that interact repeatedly should move
    IDLE → BIRTH → REINFORCE. Edges with long gaps should move toward DECAY.
    This is encoded as a soft target based on intensity and time gap.
    """
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        in_dim = feat_dim + 1 + 5          # feat | log_Δt | current_state_dist
        self.trans_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 5)
        )
        self.intensity_fc = nn.Linear(1, 1)
        self.ctx_encoder  = nn.Linear(8, hidden)
        # This loss will be computed outside in the main model (v2)
        # but kept here for the base model's logic.
        self.last_transition_loss = torch.tensor(0.0)

    def forward(self, feat: Tensor, salience: Tensor,
                delta_t: Tensor, edge_states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
          - new_edge_states (B,8)
          - edge_ctx (B, hidden)
          - student_logits (B, 5): Logits from the transition network.
          - heuristic_target_dist (B, 5): Soft targets from heuristics.
        """
        cur_dist    = torch.softmax(edge_states[:, :5], dim=-1)   # (B,5)
        log_dt      = torch.log1p(delta_t.float()).unsqueeze(-1)   # (B,1)
        intensity   = edge_states[:, 5:6]                          # (B,1)
        trans_in    = torch.cat([feat, log_dt, cur_dist], dim=-1)
        next_logits = self.trans_net(trans_in)                     # (B,5)

        # Intensity-driven transition bias:
        # High intensity → bias toward BIRTH(1)/REINFORCE(2)
        # High delta_t → bias toward DECAY(3)
        int_bias = torch.zeros_like(next_logits)
        int_bias[:, BIRTH]     = intensity.squeeze(-1) * 0.5       # more interactions → BIRTH
        int_bias[:, REINFORCE] = intensity.squeeze(-1) * 1.0       # strongly → REINFORCE
        int_bias[:, DECAY]     = log_dt.squeeze(-1) * 0.3          # long gap → DECAY
        int_bias[:, IDLE]      = -intensity.squeeze(-1) * 0.3      # less likely to stay IDLE
        student_logits = next_logits + int_bias

        # Causal masking
        cur_idx  = cur_dist.argmax(-1)                             # (B,)  on device
        mask     = VALID_TRANSITIONS[cur_idx.cpu()].to(feat.device)  # (B,5)
        masked_logits = student_logits.masked_fill(~mask, -1e9)
        new_dist = torch.softmax(masked_logits, dim=-1)            # (B,5)

        # Update intensity, volatility, lifecycle
        new_intens = (intensity
                      + torch.sigmoid(self.intensity_fc(salience.unsqueeze(-1))) * 0.1)
        changed    = (new_dist.argmax(-1) != cur_idx).float().unsqueeze(-1)
        new_vol    = edge_states[:, 6:7] * 0.9 + changed * 0.1
        new_life   = edge_states[:, 7:8] + log_dt * 0.01

        new_states = torch.cat([new_dist, new_intens, new_vol, new_life], dim=-1)
        edge_ctx   = self.ctx_encoder(new_states)

        # Transition supervision: compute soft target based on intensity & time gap
        # High intensity + short gap → should be REINFORCE
        # Low intensity + short gap → should be BIRTH
        # Any intensity + long gap → should be DECAY
        # New edge (intensity~0) → should be BIRTH
        int_val = new_intens.squeeze(-1)                               # (B,)
        dt_val = log_dt.squeeze(-1)                                     # (B,)
        target_logits = torch.zeros_like(new_dist)
        target_logits[:, IDLE]      = -int_val * 2.0                    # less IDLE as intensity grows
        target_logits[:, BIRTH]     = torch.clamp(1.0 - int_val, 0, 2) # BIRTH for new edges
        target_logits[:, REINFORCE] = int_val * 1.5                     # REINFORCE for frequent edges
        target_logits[:, DECAY]     = dt_val * 0.8                      # DECAY for long gaps
        target_logits[:, DEATH]     = dt_val * int_val.clamp(max=1) * 0.3  # DEATH for long-gap + low recent
        # Apply same causal mask to targets
        target_masked = target_logits.masked_fill(~mask, -1e9)
        heuristic_target_dist = torch.softmax(target_masked, dim=-1).detach()

        # For the base model, compute loss directly. For v2, this is done outside.
        self.last_transition_loss = F.kl_div(
            F.log_softmax(masked_logits, dim=-1),
            heuristic_target_dist,
            reduction='batchmean'
        ) * 0.1

        return new_states, edge_ctx, student_logits, heuristic_target_dist


# ────────────────────────────────────────────────────────────
# Layer 3 — TIP + DRGC
# ────────────────────────────────────────────────────────────

class TIP(nn.Module):
    """
    Temporal Interaction Parsimony — 'The Causal Hourglass'.

    Properties (from CTGC_Introduction slide 12):
      1. Impromptu Interaction elimination  (resonance gate)
      2. Zero-shot OOD robustness           (retain cause-effect soul)
      3. Resonance Detection                (high-intensity relationships)

    Mechanism: VAE-style bottleneck on node embeddings.
        min β·KL[q(z|h)‖p(z)]   s.t.  max I(z; y)
    """
    def __init__(self, hidden: int, beta: float = 0.001):
        super().__init__()
        self.beta   = beta
        self.mu_net = nn.Linear(hidden, hidden)
        self.lv_net = nn.Linear(hidden, hidden)
        # resonance gate: intensity → keep-probability for edge messages
        self.res_gate = nn.Sequential(
            nn.Linear(1, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1), nn.Sigmoid()
        )

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """h: (N, hidden) → h_parsed (N, hidden),  kl_loss scalar"""
        mu  = self.mu_net(h)
        lv  = self.lv_net(h).clamp(-4, 4)
        if self.training:
            h_p = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        else:
            h_p = mu
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean() * self.beta
        return h_p, kl

    def resonance_mask(self, intensity: Tensor) -> Tensor:
        """(B,) → (B,)  soft gate in [0,1]."""
        return self.res_gate(intensity.unsqueeze(-1)).squeeze(-1)


class DRGC(nn.Module):
    """
    Dual-Reasoning Graph Core — TxGNN++ + TIP.
    Event-triggered GRU update + TIP compression.
    """
    def __init__(self, feat_dim: int, hidden: int, beta: float = 0.001):
        super().__init__()
        self.hidden   = hidden
        self.time_enc = TimeEncoder(hidden)
        msg_in        = hidden * 3 + hidden + feat_dim   # h_s, h_d, edge_ctx, t_enc, feat
        self.msg_fn   = nn.Sequential(
            nn.Linear(msg_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.gru      = nn.GRUCell(hidden, hidden)
        self.tip      = TIP(hidden, beta)

    def forward(self, h_src: Tensor, h_dst: Tensor,
                feat: Tensor, edge_ctx: Tensor,
                dt_src: Tensor, intensity: Tensor,
                compress_all: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        h_src, h_dst   : (B, H)
        compress_all   : (N, H)  — all active node memories for TIP
        Returns: new_h_src, new_h_dst, tip_loss
        """
        t_enc = self.time_enc(dt_src)                              # (B, H)
        r_mask = self.tip.resonance_mask(intensity)                # (B,)
        msg_in = torch.cat([h_src, h_dst, edge_ctx, t_enc, feat], -1)
        msg    = self.msg_fn(msg_in) * r_mask.unsqueeze(-1)        # (B, H)

        new_h_src = self.gru(msg, h_src)
        new_h_dst = self.gru(msg, h_dst)

        # TIP on active memories
        parsed, kl = self.tip(compress_all)

        return new_h_src, new_h_dst, parsed, kl


# ────────────────────────────────────────────────────────────
# Layer 4 — NSCP
# ────────────────────────────────────────────────────────────

class NSCP(nn.Module):
    """Neuro-Symbolic Causal Policy — transition masking + CCS."""
    def __init__(self, hidden: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1)
        )

    def forward(self, src_emb: Tensor, dst_emb: Tensor,
                neg_dst_emb: Tensor, edge_state: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pos_score = self.predictor(torch.cat([src_emb, dst_emb],     -1)).squeeze(-1)
        neg_score = self.predictor(torch.cat([src_emb, neg_dst_emb], -1)).squeeze(-1)

        # Causal constraint: penalise positive predictions on DEAD edges
        death_prob  = edge_state[:, DEATH]                           # (B,)
        causal_loss = (death_prob * torch.sigmoid(pos_score)).mean()
        ccs         = 1.0 - (death_prob * torch.sigmoid(pos_score).detach())
        return pos_score, neg_score, causal_loss, ccs


# ────────────────────────────────────────────────────────────
# Full SR-GNN
# ────────────────────────────────────────────────────────────

class SRGNN(nn.Module):
    def __init__(self, num_nodes: int, feat_dim: int, hidden: int = 128,
                 tip_beta: float = 0.001,
                 lambda_tip: float = 0.01, lambda_causal: float = 0.1,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_nodes      = num_nodes
        self.feat_dim       = feat_dim
        self.hidden         = hidden
        self.lambda_tip     = lambda_tip
        self.lambda_causal  = lambda_causal
        self.device         = device
        self._feat_in       = max(feat_dim, 1)

        self.csn  = CSN(self._feat_in, hidden)
        self.ectg = ECTG(self._feat_in, hidden)
        self.drgc = DRGC(self._feat_in, hidden, tip_beta)
        self.nscp = NSCP(hidden)

        # Memory stores (tensor-based)
        self.node_mem  = NodeMemoryStore(num_nodes, hidden, device)
        self.edge_mem  = EdgeStateStore(num_nodes, hidden, device)

    def reset(self):
        self.node_mem.reset()
        self.edge_mem.reset()

    def forward(self, src: Tensor, dst: Tensor, t: Tensor,
                feat: Tensor, neg_dst: Tensor,
                rel_type: Optional[Tensor] = None) -> Dict[str, Tensor]:

        device = self.device
        B      = src.size(0)

        # Pad features
        if feat.shape[-1] == 0:
            feat = torch.zeros(B, 1, device=device)
        elif feat.shape[-1] < self._feat_in:
            feat = F.pad(feat, (0, self._feat_in - feat.shape[-1]))

        # ── L1: CSN ──────────────────────────────────────────
        dt_src    = self.node_mem.delta_t(src, t)
        feat_g, sal = self.csn(feat, dt_src)

        # ── L2: ECTG ─────────────────────────────────────────
        edge_st   = self.edge_mem.get_batch(src, dst)
        new_est, edge_ctx, _, _ = self.ectg(feat_g, sal, dt_src, edge_st)
        self.edge_mem.update_batch(src, dst, new_est)
        intensity = new_est[:, 5]                                   # (B,)

        # ── L3: DRGC + TIP ───────────────────────────────────
        h_src     = self.node_mem.get(src)
        h_dst     = self.node_mem.get(dst)
        dt_dst    = self.node_mem.delta_t(dst, t)

        # TIP compresses all active node memories
        all_idx   = torch.unique(torch.cat([src, dst]))
        all_h     = self.node_mem.get(all_idx)

        new_h_src, new_h_dst, parsed_h, kl = self.drgc(
            h_src, h_dst, feat_g, edge_ctx, dt_src, intensity, all_h
        )

        # ── L4: NSCP — score BEFORE memory update (prevent leakage) ──
        neg_dst_emb = self.node_mem.get(neg_dst)

        pos_sc, neg_sc, c_loss, ccs = self.nscp(
            new_h_src, new_h_dst, neg_dst_emb, new_est
        )

        # NOW update memory (after scoring)
        self.node_mem.set(all_idx, parsed_h)
        self.node_mem.set(src, new_h_src)
        self.node_mem.set(dst, new_h_dst)
        self.node_mem.update_time(torch.cat([src, dst]), torch.cat([t, t]))

        # ── Loss ─────────────────────────────────────────────
        pred_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_sc, neg_sc]),
            torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])
        )
        trans_loss = self.ectg.last_transition_loss
        total_loss = pred_loss + self.lambda_tip * kl + self.lambda_causal * c_loss + trans_loss

        return {
            "pos_score":    pos_sc,
            "neg_score":    neg_sc,
            "loss":         total_loss,
            "pred_loss":    pred_loss,
            "trans_loss":   trans_loss,
            "tip_loss":     kl,
            "causal_loss":  c_loss,
            "ccs":          ccs.mean(),
            "salience":     sal.mean(),
        }

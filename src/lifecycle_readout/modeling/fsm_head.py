"""
FSM Head for RS-GNN v3.3.

Components:
  - StateObserver: reads h.detach() → soft state distribution s_t
  - TransitionPredictor: (h, s_t) → 5x5 transition logits
  - LifecycleFSMMask: soft mask with ever_alive accumulator
  - ExistenceDecoder: P(s_{t+1}) → P(edge exists)

Key design: ALL FSM components use h.detach() — never propagate gradient to backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

IDLE, BIRTH, REINFORCE, DECAY, DEATH = 0, 1, 2, 3, 4


PRIOR_LIFECYCLE = torch.tensor([
    #  IDLE  BIRTH  REINF  DECAY  DEATH
    [ 1.0,  1.0,  -3.0,  -3.0,  -5.0],  # from IDLE
    [-1.0,  1.0,   1.0,  -1.0,  -3.0],  # from BIRTH
    [-1.0, -1.0,   1.0,   1.0,  -3.0],  # from REINFORCE
    [-1.0, -1.0,   0.0,   1.0,   1.0],  # from DECAY
    [-1.0,  0.0,  -3.0,  -3.0,   1.0],  # from DEATH (revival possible but discouraged)
], dtype=torch.float32)


# ── HARD causal-rule admissibility matrix C ∈ {0,1}^{5×5} (the actual LFG core) ──
# C[i, j] == 1  iff the lifecycle transition  argmax(s_t)=i → argmax(s_{t+1})=j  is
# CAUSALLY ADMISSIBLE; 0 iff it is causally IMPOSSIBLE. This is a HARD {0,1} rule
# matrix (NOT the soft sigmoid(prior+delta) FSM mask) used only to derive a per-event
# gradient gate — it never enters the prediction value.
#
# RECONCILED 2026-05-31 with VALID_TRANSITIONS (sr_gnn.py:27) so the hard gate's
# "violation" definition is IDENTICAL to the ECTG causal mask: C == VALID_TRANSITIONS
# as {0,1}. Both now encode the same TRIDIAGONAL / bi-directional admissibility —
# each state reaches {previous, self, next}, single-rung moves only. The prior C was
# a different (non-adjacent) rule set (e.g. it allowed BIRTH→IDLE / DEATH→BIRTH but
# forbade BIRTH→BIRTH... actually allowed self), which meant the LFG gate flagged a
# DIFFERENT set of transitions as violations than the mask used for prediction — an
# inconsistency. Encoded rules (single-rung, no 2-step jumps):
#   - IDLE (no prev): only stay IDLE or be BORN; cannot skip to REINFORCE/DECAY/DEATH.
#   - BIRTH: {IDLE (dormancy), BIRTH, REINFORCE}; not DECAY/DEATH (no 2-step).
#   - REINFORCE: {BIRTH (regress), REINFORCE, DECAY}; not IDLE/DEATH directly.
#   - DECAY: {REINFORCE (recovery), DECAY, DEATH}; not IDLE/BIRTH directly.
#   - DEATH: {DECAY (single-rung revival path), DEATH}; revival now flows back through
#     DECAY→REINFORCE→… one rung at a time (matches the bi-directional FSM), not a
#     direct DEATH→BIRTH jump.
CAUSAL_RULE_MATRIX = torch.tensor([
    #  IDLE BIRTH REINF DECAY DEATH        (to →)
    [   1,   1,    0,    0,    0  ],  # from IDLE       : no prev → stay / be born
    [   1,   1,    1,    0,    0  ],  # from BIRTH      : dormant / stay / reinforce
    [   0,   1,    1,    1,    0  ],  # from REINFORCE  : regress / stay / decay
    [   0,   0,    1,    1,    1  ],  # from DECAY      : recover / stay / DIE
    [   0,   0,    0,    1,    1  ],  # from DEATH      : revive-via-DECAY / stay dead
], dtype=torch.float32)


def compute_causal_validity(s_t: Tensor, s_t1: Tensor, C: Tensor) -> Tensor:
    """Per-event causal validity v_e ∈ {0,1} from the HARD rule matrix C.

    The FSM transition is read as a *hard* argmax of the (already detached) soft
    distributions: i = argmax(s_t), j = argmax(s_{t+1}); v_e = C[i, j].
    Inputs are detached defensively so this can ONLY produce a gradient *mask*,
    never its own gradient. Returns (B,) float in {0,1}.
    """
    with torch.no_grad():
        i = s_t.detach().argmax(-1)
        j = s_t1.detach().argmax(-1)
        v = C[i, j]
    return v


class StateObserver(nn.Module):
    """Read continuous h → soft state distribution. No backprop to h."""
    def __init__(self, hidden: int, n_states: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_states),
        )

    def forward(self, h: Tensor, detach_h: bool = True) -> Tensor:
        """h: (B, hidden) → s: (B, 5) softmax distribution.

        detach_h (default True = byte-identical canonical "No backprop to h"):
        when False, gradient flows through h into the backbone — used ONLY by the
        single-variable detach probe on the AP-scoring path (edge_h_detach_scorepath
        =False). Every other call (FSM-stream / interpretation) keeps detach_h=True.
        """
        return torch.softmax(self.net(h if not detach_h else h.detach()), dim=-1)


class TransitionPredictor(nn.Module):
    """Predict P(s_{t+1} | s_t, h). Low-rank parameterization.

    pair_phi_dim (fsm_arch="v3" REDESIGN, default 0 = CANONICAL / v1 / v2 byte-
    identical): when >0, an ADDITIONAL per-pair operator g(φ_uv) is built. φ_uv is a
    small vector of CONTINUOUS per-pair history statistics already accumulated in
    EdgeStateStoreV3 / ECTGv3 (Hawkes λ, Welford inter-event mean/var, recurrence
    EWMA, staleness, ever_alive — see sr_gnn_v3_3.py forward where φ is assembled).
    A 2-layer MLP maps φ_uv → a full n_states×n_states bias matrix that is ADDED to
    the global low-rank operator T = U@V BEFORE applying s_t:

        T_uv(i→j) = (U@V)(i,j) [global, h-conditioned] + g(φ_uv)(i,j) [per-pair]

    so the flip dynamics of EACH interacting pair are derived from that pair's OWN
    continuous past. With pair_phi_dim==0 the g-branch is not constructed (zero extra
    params) and forward is bit-identical to the original.
    """
    def __init__(self, hidden: int, n_states: int = 5, rank: int = 3,
                 pair_phi_dim: int = 0):
        super().__init__()
        self.n_states = n_states
        self.rank     = rank
        self.pair_phi_dim = pair_phi_dim
        # h → U (B, n_states, rank)
        self.U_net = nn.Linear(hidden, n_states * rank)
        # h → V (B, rank, n_states)
        self.V_net = nn.Linear(hidden, rank * n_states)
        # ── Per-pair operator g(φ_uv) → (n_states*n_states) bias (fsm_arch="v3") ──
        if pair_phi_dim > 0:
            phi_hidden = max(8, 2 * pair_phi_dim)
            self.pair_g = nn.Sequential(
                nn.Linear(pair_phi_dim, phi_hidden), nn.ReLU(),
                nn.Linear(phi_hidden, n_states * n_states),
            )
            # zero-init the output layer so a FRESH model starts == global operator
            # (g(φ)=0); the per-pair bias is LEARNED away from zero by supervision.
            nn.init.zeros_(self.pair_g[-1].weight)
            nn.init.zeros_(self.pair_g[-1].bias)
        else:
            self.pair_g = None

    def forward(self, h: Tensor, s_t: Tensor,
                pair_phi: Tensor = None, return_T: bool = False):
        """
        h:        (B, hidden)
        s_t:      (B, n_states)
        pair_phi: (B, pair_phi_dim) per-pair history stats (fsm_arch="v3" only)
        Returns trans_logits: (B, n_states); if return_T also the per-pair operator
        T_uv (B, n_states, n_states) for the heterogeneity diagnostic.
        """
        B = h.size(0)
        U = self.U_net(h).view(B, self.n_states, self.rank)
        V = self.V_net(h).view(B, self.rank, self.n_states)
        # T = U @ V → (B, n_states, n_states), T[i,j] = logit of going from state i to j
        T = torch.bmm(U, V)
        if self.pair_g is not None and pair_phi is not None:
            # per-pair additive operator bias from THIS pair's continuous history
            g = self.pair_g(pair_phi).view(B, self.n_states, self.n_states)
            T = T + g
        # Apply with current state: s_{t+1}_logits = s_t @ T
        next_logits = torch.bmm(s_t.unsqueeze(1), T).squeeze(1)
        if return_T:
            return next_logits, T
        return next_logits


class LifecycleFSMMask(nn.Module):
    """
    Soft, learnable lifecycle mask with ever_alive accumulator.
    Each edge has ever_alive ∈ [0,1] tracking whether it's ever been BIRTH/REINFORCE.
    """
    def __init__(self, n_states: int = 5):
        super().__init__()
        self.n_states = n_states
        # learnable adjustment to prior
        self.delta = nn.Parameter(torch.zeros(n_states, n_states))
        self.register_buffer("prior", PRIOR_LIFECYCLE.clone())

    def get_mask_from_state(self, s_t: Tensor) -> Tensor:
        """
        s_t: (B, n_states) soft current state
        Returns: (B, n_states) — mask for next state
        """
        # Soft current state index via expectation: each row of mask weighted by s_t
        # mask[i, j] = prob of allowed transition i → j
        full_mask = torch.sigmoid(self.prior + self.delta)  # (5, 5)
        # Per-event mask: M_b[j] = sum_i s_t[b, i] * full_mask[i, j]
        per_event_mask = torch.einsum("bi,ij->bj", s_t, full_mask)
        return per_event_mask

    def apply_ever_alive_gate(self, next_dist: Tensor, ever_alive: Tensor) -> Tensor:
        """
        next_dist: (B, 5)
        ever_alive: (B,) ∈ [0, 1]
        Block DEATH if never alive.
        """
        gate = ever_alive.unsqueeze(-1)  # (B, 1)
        # death_mask: 1 if ever alive, small otherwise
        block_death = torch.zeros_like(next_dist)
        block_death[:, DEATH] = (1 - gate.squeeze(-1)) * (-5.0)  # subtract 5 from DEATH if never alive
        return next_dist + block_death


class ExistenceDecoder(nn.Module):
    """P(edge) = weighted sum of P(s_{t+1}).

    fix_existence_init (default False = CANONICAL, unchanged):
      False → theta = init.log().clamp(min=-3). This is the ORIGINAL canonical
        init; because the forward uses softplus(theta) (not exp), the EFFECTIVE
        weights are softplus(log(x)) ≈ [0.095, 0.693, 0.693, 0.262, 0.049],
        which deviates from the intended spec [0.1, 1, 1, 0.3, 0].
      True  → theta = softplus_inverse(init) = log(exp(x) - 1), so that
        softplus(theta) == init EXACTLY = [0.1, 1, 1, 0.3, 0] (the spec). The
        x == 0 entry (DEATH) maps to softplus_inverse(0) = -inf, replaced by a
        large-negative -10.0 so softplus(-10) ≈ 4.5e-5 ≈ 0.
    """
    def __init__(self, n_states: int = 5, fix_existence_init: bool = False):
        super().__init__()
        # init: BIRTH and REINFORCE weight high
        init = torch.tensor([0.1, 1.0, 1.0, 0.3, 0.0])
        if fix_existence_init:
            # softplus-inverse so softplus(theta) == init exactly; x=0 → -10
            theta0 = torch.where(
                init > 0,
                torch.log(torch.expm1(init.clamp(min=1e-6))),
                torch.full_like(init, -10.0),
            )
            self.theta = nn.Parameter(theta0)  # log-space (softplus-inverse)
        else:
            self.theta = nn.Parameter(init.log().clamp(min=-3))  # log-space (canonical)

    def forward(self, next_state_dist: Tensor) -> Tensor:
        """
        next_state_dist: (B, 5)
        Returns: (B,) edge existence logit
        """
        w = F.softplus(self.theta)
        p_edge = (w.unsqueeze(0) * next_state_dist).sum(-1)
        # Convert to logit: log(p / (1 - p))
        p_clamped = p_edge.clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p_clamped / (1 - p_clamped))
        return logit


def compute_compliance(s_t: Tensor, s_t1: Tensor,
                       ever_alive: Tensor, hawkes_lam: Tensor) -> Tensor:
    """
    Compute LFG compliance score per event.
    s_t, s_t1: (B, 5) soft state distributions
    ever_alive: (B,) ∈ [0, 1]
    hawkes_lam: (B,) hawkes intensity
    Returns: c ∈ [0.05, 1.0] per event, detached.
    """
    with torch.no_grad():
        # Rule 1 (HARD): cannot DEATH if never alive
        rule1 = 1.0 - s_t1[:, DEATH] * (1 - ever_alive)

        # Rule 2 (SOFT): smooth transitions (small change preferred)
        diff = (s_t1 - s_t).abs().sum(-1)
        rule2 = torch.exp(-3.0 * diff)

        # Rule 3 (SOFT): Hawkes-consistency
        lam_norm = hawkes_lam / (hawkes_lam.mean() + 1e-6)
        expected_active = lam_norm / (lam_norm + 1.0)  # (B,)
        # If high λ → expect REINFORCE+BIRTH; low λ → IDLE+DEATH
        active_score = s_t1[:, BIRTH] + s_t1[:, REINFORCE]
        rule3 = 1.0 - (active_score - expected_active).abs()
        rule3 = rule3.clamp(min=0.0, max=1.0)

        compliance = (rule1 * rule2 * rule3).clamp(min=0.05, max=1.0)
    return compliance

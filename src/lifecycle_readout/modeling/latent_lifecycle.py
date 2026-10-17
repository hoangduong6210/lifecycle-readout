"""Latent phase encoder, silence flow and candidate-time scoring transform.

Phase zero is a designated dormant anchor, not an identified relationship state.
Ordered gate weights impose non-increase on the exact pure-silence flow at fixed
initial state and affinity, subject to the numerical boundary. The scored
candidate transform can reverse that response, even with a constant transition
matrix; see tests/test_silence_scope.py and the canonical method notes.
Scoring through z_next permits task-gradient paths to the decay and transition
parameters. Connected paths need not yield nonzero gradients at every input.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LatentLifecycleConfig:
    n_phases: int = 5
    hidden_dim: int = 128
    phi_dim: int = 6
    temperature: float = 1.0
    min_temperature: float = 0.2
    decay_init: float = 0.1
    lambda_entropy: float = 0.01
    lambda_diversity: float = 0.05
    lambda_sparse: float = 0.001


class LatentPhaseEncoder(nn.Module):
    """Encodes edge representation h_uv and dynamic history stats phi_uv into phase distribution z in Delta^{K-1}."""

    def __init__(self, hidden_dim: int, phi_dim: int, n_phases: int = 5):
        super().__init__()
        self.n_phases = n_phases
        in_dim = hidden_dim + phi_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_phases),
        )

    def forward(
        self,
        h: Tensor,
        phi: Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tensor:
        """h: (B, hidden), phi: (B, phi_dim) -> z: (B, n_phases) in simplex Delta^{K-1}."""
        x = torch.cat([h, phi], dim=-1)
        logits = self.net(x)
        if self.training:
            return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        return F.softmax(logits / max(temperature, 1e-4), dim=-1)


class ContinuousTransitionKernel(nn.Module):
    """Continuous-time jump-decay transition dynamics.

    In the absence of interaction, active phases k in {1, ..., K-1} decay exponentially toward
    dormant phase 0 with rates d_k = softplus(theta_{d, k}) > 0.
    At candidate evaluation, an unconstrained dynamic operator T_theta transforms the
    decayed state. This transform is evaluated before an event label is observed and
    is not covered by the pure-silence monotonicity guarantee.
    """

    def __init__(self, phi_dim: int, n_phases: int = 5, decay_init: float = 0.1):
        super().__init__()
        self.n_phases = n_phases
        # Non-negative decay rates d_1, ..., d_{K-1} for active phases
        init_raw = math.log(math.exp(decay_init) - 1.0) if decay_init > 0 else -2.0
        self.decay_param = nn.Parameter(torch.full((n_phases - 1,), init_raw))

        # Dynamic transition operator T_theta(phi, log(1 + dt))
        self.trans_net = nn.Sequential(
            nn.Linear(phi_dim + 1, n_phases * 4),
            nn.GELU(),
            nn.Linear(n_phases * 4, n_phases * n_phases),
        )
        # Initialize transition weights small so initial dynamics respect continuous decay
        nn.init.xavier_uniform_(self.trans_net[0].weight)
        nn.init.normal_(self.trans_net[-1].weight, std=0.01)
        nn.init.zeros_(self.trans_net[-1].bias)

    @property
    def decay_rates(self) -> Tensor:
        """Decay rates for active phases k >= 1 (strictly non-negative via softplus)."""
        return F.softplus(self.decay_param)

    def continuous_decay(self, z_t: Tensor, dt: Tensor) -> Tensor:
        """Decays active phases toward dormant phase 0 over silence interval dt.

        z_k(dt) = z_k(0) * exp(-d_k * dt) for k >= 1.
        z_0(dt) = 1 - sum_{k >= 1} z_k(dt).
        """
        B = z_t.size(0)
        dt_col = dt.view(B, 1).clamp(min=0.0)
        active_z = z_t[:, 1:]  # (B, K-1)
        decay_factor = torch.exp(-self.decay_rates.unsqueeze(0) * dt_col)  # (B, K-1)
        decayed_active = active_z * decay_factor  # (B, K-1)
        decayed_dormant = (1.0 - decayed_active.sum(dim=-1, keepdim=True)).clamp(min=1e-6, max=1.0)
        z_decayed = torch.cat([decayed_dormant, decayed_active], dim=-1)
        # Renormalize for numerical stability on simplex
        return z_decayed / z_decayed.sum(dim=-1, keepdim=True)

    def forward(
        self,
        z_t: Tensor,
        phi: Tensor,
        dt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """z_t: (B, K), phi: (B, phi_dim), dt: (B,) or (B, 1) -> (z_next, T_matrix).

        Propagates state across silence dt, then applies event jump operator T_theta.
        """
        B = z_t.size(0)
        dt_col = dt.view(B, 1).clamp(min=0.0)
        log_dt = torch.log1p(dt_col)

        # 1. Continuous decay along silence interval dt
        z_decayed = self.continuous_decay(z_t, dt)

        # 2. Dynamic event transition operator T in R^{K x K}
        inp = torch.cat([phi, log_dt], dim=-1)
        T = self.trans_net(inp).view(B, self.n_phases, self.n_phases)

        # Zero diagonal to prevent self-loops in transition matrix
        eye = torch.eye(self.n_phases, device=z_t.device).unsqueeze(0)
        T = T * (1.0 - eye)

        # 3. State update combining continuous decay with dynamic transition jump
        eps = 1e-6
        log_z = torch.log(z_decayed.clamp(min=eps))
        jump_logits = log_z + torch.bmm(z_decayed.unsqueeze(1), T).squeeze(1)
        z_next = F.softmax(jump_logits, dim=-1)

        return z_next, T


class LifecycleSurvivalGate(nn.Module):
    """Ordered sigmoid attenuation factor Psi(z) in (0, 1).

    Active weights exceed the dormant weight by cumulative softplus increments.
    At fixed initial state, exact pure decay decreases this factor when active
    mass is positive, in finite-time exact arithmetic. A preceding candidate
    transform, changing context or changing base score is outside this guarantee.
    This factor is not a fitted time-to-event survival distribution.
    """

    def __init__(self, n_phases: int = 5):
        super().__init__()
        self.n_phases = n_phases
        self.w_base = nn.Parameter(torch.tensor(-2.0))
        self.deltas = nn.Parameter(torch.full((n_phases - 1,), 0.5))
        self.b_surv = nn.Parameter(torch.zeros(1))

    @property
    def survival_weights(self) -> Tensor:
        """Returns ordered weights [w_0, w_1, ..., w_{K-1}] with w_0 < w_1 < ..."""
        increments = F.softplus(self.deltas)
        cum_increments = torch.cumsum(increments, dim=0)
        active_weights = self.w_base + cum_increments
        return torch.cat([self.w_base.view(1), active_weights])

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, K) -> psi: (B,) survival factor in (0, 1)."""
        w = self.survival_weights
        logits = torch.matmul(z, w) + self.b_surv
        return torch.sigmoid(logits)


class UnconstrainedSurvivalGate(nn.Module):
    """Ablation: unconstrained linear survival gate w in R^K without forced ordering.

    Used to empirically test whether data-driven training naturally discovers
    monotonic ordering without architectural constraints.
    """

    def __init__(self, n_phases: int = 5):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_phases) * 0.1)
        self.b_surv = nn.Parameter(torch.zeros(1))

    @property
    def survival_weights(self) -> Tensor:
        return self.w

    def forward(self, z: Tensor) -> Tensor:
        logits = torch.matmul(z, self.w) + self.b_surv
        return torch.sigmoid(logits)


class LatentLifecycleModule(nn.Module):
    """Latent phase candidate scorer with connected task-gradient paths.

    Scoring z_next permits gradients to reach decay and transition parameters;
    it guarantees neither nonzero derivatives nor a monotone complete score.
    """

    def __init__(self, cfg: Optional[LatentLifecycleConfig] = None, unconstrained: bool = False):
        super().__init__()
        self.cfg = cfg or LatentLifecycleConfig()
        self.unconstrained = unconstrained
        self.encoder = LatentPhaseEncoder(
            hidden_dim=self.cfg.hidden_dim,
            phi_dim=self.cfg.phi_dim,
            n_phases=self.cfg.n_phases,
        )
        self.kernel = ContinuousTransitionKernel(
            phi_dim=self.cfg.phi_dim,
            n_phases=self.cfg.n_phases,
            decay_init=self.cfg.decay_init,
        )
        if unconstrained:
            self.survival_gate = UnconstrainedSurvivalGate(n_phases=self.cfg.n_phases)
        else:
            self.survival_gate = LifecycleSurvivalGate(n_phases=self.cfg.n_phases)

    def forward(
        self,
        h_uv: Tensor,
        phi_uv: Tensor,
        dt: Tensor,
        base_logits: Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, Tensor]:
        """h_uv: (B, hidden), phi_uv: (B, phi_dim), dt: (B,), base_logits: (B,)."""
        # 1. Encoded latent phase distribution from candidate inputs
        z_curr = self.encoder(h_uv, phi_uv, temperature=temperature)

        # 2. Propagate through continuous-time jump-decay transition kernel across dt
        z_next, T_matrix = self.kernel(z_curr, phi_uv, dt)

        # 3. Survival gating factor evaluated on z_next (the state at edge prediction time)
        psi = self.survival_gate(z_next)

        # 4. Sampled-candidate score: sigma(base_logits) * psi(z_next)
        base_prob = torch.sigmoid(base_logits)
        gated_prob = base_prob * psi
        gated_prob_clamped = gated_prob.clamp(1e-6, 1.0 - 1e-6)
        gated_logits = torch.log(gated_prob_clamped / (1.0 - gated_prob_clamped))

        return {
            "z_curr": z_curr,
            "z_next": z_next,
            "T_matrix": T_matrix,
            "survival_gate": psi,
            "base_prob": base_prob,
            "gated_prob": gated_prob,
            "gated_logits": gated_logits,
        }

    def counterfactual_trajectory(
        self,
        z_curr: Tensor,
        base_logits: Tensor,
        dt_grid: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Evaluate pure-silence grid scores at fixed state and affinity.

        The candidate-time transition and encoder are not reevaluated here.
        Grid agreement checks this diagnostic, not every point or the full score.
        """
        if dt_grid is None:
            dt_grid = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

        B = z_curr.size(0)
        base_prob = torch.sigmoid(base_logits)
        prob_steps = []

        for dt_val in dt_grid:
            dt_tensor = torch.full((B,), dt_val, device=z_curr.device, dtype=torch.float32)
            # Silence intervention: pure continuous decay without new interaction jump
            z_decayed = self.kernel.continuous_decay(z_curr, dt_tensor)
            psi = self.survival_gate(z_decayed)
            prob_steps.append(base_prob * psi)

        prob_trajectory = torch.stack(prob_steps, dim=-1)  # (B, num_steps)
        # Check step-by-step non-increasing: prob[step+1] <= prob[step] + eps
        diffs = prob_trajectory[:, 1:] - prob_trajectory[:, :-1]
        is_monotonic = (diffs <= 1e-5).all(dim=-1)

        # Terminal silence effect at max dt
        delta_prob_silence = prob_trajectory[:, -1] - prob_trajectory[:, 0]

        return {
            "prob_trajectory": prob_trajectory,
            "is_monotonic": is_monotonic,
            "monotonic_rate": is_monotonic.float().mean(),
            "silence_treatment_effect": delta_prob_silence.mean(),
        }

    def phase_intervention_ate(
        self,
        base_logits: Tensor,
        k: int,
        k_ref: int = 0,
    ) -> Tensor:
        """Computes true causal phase intervention ATE(k -> k_ref) as defined in wiki:

        ATE(k -> k_ref) = E[ y_hat(do(z = e_k)) - y_hat(do(z = e_{k_ref})) ]
        """
        B = base_logits.size(0)
        e_k = torch.zeros(B, self.cfg.n_phases, device=base_logits.device)
        e_k[:, k] = 1.0
        e_ref = torch.zeros(B, self.cfg.n_phases, device=base_logits.device)
        e_ref[:, k_ref] = 1.0

        base_prob = torch.sigmoid(base_logits)
        psi_k = self.survival_gate(e_k)
        psi_ref = self.survival_gate(e_ref)

        effect = (base_prob * psi_k) - (base_prob * psi_ref)
        return effect.mean()

    def compute_losses(
        self,
        out: Dict[str, Tensor],
        targets: Tensor,
        z_target_future: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Computes joint link prediction BCE, transition consistency, and information bottleneck regularizers."""
        # 1. Main Link Prediction Task Loss (flows gradients through survival gate, z_next, decay, and T)
        bce_loss = F.binary_cross_entropy_with_logits(out["gated_logits"], targets.float())

        # 2. Transition Consistency Loss (if future state target is provided)
        if z_target_future is not None:
            trans_loss = F.kl_div(
                torch.log(out["z_next"].clamp(min=1e-6)),
                z_target_future,
                reduction="batchmean",
            )
        else:
            trans_loss = torch.tensor(0.0, device=targets.device)

        # 3. Information Bottleneck Regularizers:
        # a) Individual phase entropy (minimization -> sharp phase assignments)
        eps = 1e-6
        z = out["z_curr"]
        entropy = -torch.sum(z * torch.log(z + eps), dim=-1).mean()

        # b) Batch diversity entropy (maximization -> uniform phase coverage across dataset)
        z_mean = z.mean(dim=0)
        diversity = -torch.sum(z_mean * torch.log(z_mean + eps))

        # c) Transition matrix sparsity
        sparsity = out["T_matrix"].abs().mean()

        reg_loss = (
            self.cfg.lambda_entropy * entropy
            - self.cfg.lambda_diversity * diversity
            + self.cfg.lambda_sparse * sparsity
        )

        total_loss = bce_loss + trans_loss + reg_loss

        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            "trans_loss": trans_loss,
            "entropy": entropy,
            "diversity": diversity,
            "sparsity": sparsity,
            "reg_loss": reg_loss,
        }

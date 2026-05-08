"""
CFSDTM.py — Chain-Free Sparse Dynamic Topic Model
==================================================
Wraps CFDTM (Wu et al., ACL 2024) with per-time-slice topic activity gates,
implementing a differentiable analogue of the spike-and-slab selector from
SDTM (Zhou et al., Knowledge and Information Systems, 2025).

New components vs CFDTM
-----------------------
TopicGate       : nn.Module — learnable sigmoid gates g[t,k] ∈ (0,1)
sparsity_loss   : binary entropy — pushes gates toward 0 or 1
smoothness_loss : L2 penalty on Δg between adjacent time slices
_apply_gate     : masks and renormalises document-topic distributions θ

Mathematical correspondence with SDTM
--------------------------------------
SDTM:    b_k^(t) ~ Bernoulli(π_t),  π_t ~ Beta(s, v)
CF-SDTM: g[t,k]  = sigmoid(logit[t,k]),  logit trained via sparsity_loss
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Topic Gate
# ─────────────────────────────────────────────────────────────────────────────

class TopicGate(nn.Module):
    """
    Learnable activity gates g[t, k] ∈ (0, 1).

    Shape: (num_times, num_topics)

      g[t,k] → 1  :  topic k is 'alive'  at slice t  (birth)
      g[t,k] → 0  :  topic k is 'dead'   at slice t  (death)
    """

    def __init__(self, num_times: int, num_topics: int):
        super().__init__()
        # Initialised to 0  →  sigmoid(0) = 0.5  (neutral start)
        self.logits = nn.Parameter(torch.zeros(num_times, num_topics))

    def forward(self) -> torch.Tensor:
        """Gate matrix (T, K), values in (0, 1)."""
        return torch.sigmoid(self.logits)

    def sparsity_loss(self) -> torch.Tensor:
        """
        Binary entropy  H(g) = -[g·log g + (1−g)·log(1−g)].

        Maximised at g = 0.5, zero at g ∈ {0, 1}.
        Minimising this loss pushes gates to the boundaries — the same
        effect as the U-shaped Beta(0.5, 0.5) prior on π_t in SDTM.
        """
        g = self.forward()
        eps = 1e-7
        return -(g * (g + eps).log() + (1 - g) * (1 - g + eps).log()).mean()

    def smoothness_loss(self) -> torch.Tensor:
        """
        L2 penalty on temporal differences  g[t+1] − g[t].
        Encourages gradual birth/death rather than per-epoch flickering.
        """
        g = self.forward()
        if g.shape[0] <= 1:
            return g.new_tensor(0.0)
        return (g[1:] - g[:-1]).pow(2).mean()

    @torch.no_grad()
    def active_mask(self, threshold: float = 0.3) -> torch.Tensor:
        """Boolean (T, K): True where the topic is considered alive."""
        return self.forward() > threshold


# ─────────────────────────────────────────────────────────────────────────────
#  CF-SDTM
# ─────────────────────────────────────────────────────────────────────────────

class CFSDTM(nn.Module):
    """
    Chain-Free Sparse Dynamic Topic Model.

    Minimal usage (mirrors CFDTM in TopMost):

        model = CFSDTM(
            vocab_size             = dataset.vocab_size,
            num_times              = dataset.num_times,
            num_topics             = 50,
            pretrained_WE          = dataset.pretrained_WE,
            train_time_wordfreq    = dataset.train_time_wordfreq,
            lambda_sparse          = 0.1,
            lambda_smooth          = 0.05,
        )

    Extra hyperparameters
    ---------------------
    lambda_sparse   : weight for sparsity loss   (default 0.1)
    lambda_smooth   : weight for smoothness loss  (default 0.05)
    gate_threshold  : g > threshold → topic alive (default 0.3)
    """

    def __init__(
        self,
        vocab_size: int,
        num_times: int,
        num_topics: int = 50,
        pretrained_WE=None,
        train_time_wordfreq=None,
        lambda_sparse: float = 0.1,
        lambda_smooth: float = 0.05,
        gate_threshold: float = 0.3,
        **cfdtm_kwargs,
    ):
        super().__init__()

        from topmost.models.dynamic.CFDTM.CFDTM import CFDTM  # late import

        self.num_times      = num_times
        self.num_topics     = num_topics
        self.lambda_sparse  = lambda_sparse
        self.lambda_smooth  = lambda_smooth
        self.gate_threshold = gate_threshold

        # ── Base CFDTM (unchanged) ──────────────────────────────────────────
        self.base = CFDTM(
            vocab_size          = vocab_size,
            num_times           = num_times,
            num_topics          = num_topics,
            pretrained_WE       = pretrained_WE,
            train_time_wordfreq = train_time_wordfreq,
            **cfdtm_kwargs,
        )

        # ── Topic Activity Gates (new) ──────────────────────────────────────
        self.gate = TopicGate(num_times, num_topics)

    # ── Gate interface ───────────────────────────────────────────────────────

    def get_gates(self) -> torch.Tensor:
        """Returns gate matrix (T, K). Call .detach().cpu() before plotting."""
        return self.gate()

    def gate_loss(self) -> torch.Tensor:
        """Combined gate regularisation loss to add to the main ELBO."""
        return (
            self.lambda_sparse * self.gate.sparsity_loss()
            + self.lambda_smooth * self.gate.smoothness_loss()
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def _apply_gate(
        self, theta: torch.Tensor, time_ids: torch.Tensor
    ) -> torch.Tensor:
        """Mask theta by slice-specific gates, then renormalise to a distribution."""
        g      = self.gate()[time_ids]                    # (B, K)
        masked = theta * g
        return masked / masked.sum(-1, keepdim=True).clamp(min=1e-8)

    def _parse_base_output(self, base_out):
        """
        Extract loss regardless of what CFDTM.forward returns.
        CFDTM returns {'loss': loss}; this handles that plus tuple/tensor fallbacks.
        """
        if isinstance(base_out, dict):
            loss = (base_out.get('loss')
                    or base_out.get('nelbo')
                    or base_out.get('total_loss'))
        elif isinstance(base_out, (tuple, list)):
            loss = base_out[0]
        else:
            loss = base_out
        return loss

    def forward(self, x: torch.Tensor, times: torch.Tensor = None):
        """
        Parameters
        ----------
        x     : bag-of-words tensor,  shape (B, V),  dtype float
        times : time-slice indices,   shape (B,),    dtype long

        Returns
        -------
        dict with key 'loss' — base CFDTM loss + gate regularisation.
        Compatible with DynamicTrainer (expects rst_dict['loss']).
        """
        if times is not None:
            base_out = self.base(x, times)
        else:
            base_out = self.base(x)

        base_loss  = self._parse_base_output(base_out)
        total_loss = base_loss + self.gate_loss()

        return {'loss': total_loss}

    # ── TopMost-compatible helpers ───────────────────────────────────────────

    def get_theta(
        self, x: torch.Tensor, times: torch.Tensor = None
    ) -> torch.Tensor:
        """Infer gated document-topic distributions (no gradient)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            # base.get_theta returns plain theta tensor in eval mode
            theta = self.base.get_theta(x)
            result = self._apply_gate(theta, times) if times is not None else theta
        if was_training:
            self.train()
        return result

    # ── Attribute proxy ──────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        """
        Forward unknown attribute lookups to self.base so TopMost evaluation
        utilities (model.get_beta, model.topic_embeddings, etc.) keep working.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                'base_state' : self.base.state_dict(),
                'gate_logits': self.gate.logits.data.cpu(),
                'config'     : {
                    'num_times'     : self.num_times,
                    'num_topics'    : self.num_topics,
                    'lambda_sparse' : self.lambda_sparse,
                    'lambda_smooth' : self.lambda_smooth,
                    'gate_threshold': self.gate_threshold,
                },
            },
            path,
        )
        print(f'CF-SDTM checkpoint saved  →  {path}')

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        vocab_size: int,
        pretrained_WE=None,
        train_time_wordfreq=None,
        device: str = 'cpu',
    ) -> 'CFSDTM':
        ckpt  = torch.load(path, map_location=device)
        cfg   = ckpt['config']
        model = cls(
            vocab_size          = vocab_size,
            pretrained_WE       = pretrained_WE,
            train_time_wordfreq = train_time_wordfreq,
            **cfg,
        )
        model.base.load_state_dict(ckpt['base_state'])
        model.gate.logits.data = ckpt['gate_logits'].to(device)
        print(f'CF-SDTM checkpoint loaded  ←  {path}')
        return model.to(device)

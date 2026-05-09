"""
CFSDTMTrainer.py — Two-phase training loop for CF-SDTM
=======================================================

Phase 1 — Warmup (warmup_epochs):
  Gates are frozen at g≈1.  Only beta (topic embeddings) and the VAE encoder
  are trained.  After warmup, topic utilization carries a genuine signal about
  which topics are active in which time slices.

Phase 2 — Fine-tune (finetune_epochs):
  Gate logits are initialised from warmup utilization via log-odds.
  All parameters are trained jointly with three gate losses:
    sparse      — binary entropy, pushes gates to 0 or 1
    smooth      — L2 temporal penalty, discourages flickering
    supervision — MSE to warmup-derived targets, provides directional signal

  The supervision target is fixed after warmup.  Re-computing it during
  fine-tune creates a feedback loop (gates → theta → U → gates) that is
  less stable than using the one-time warmup estimate.
"""

from __future__ import annotations

import os
import time
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from typing import Optional

from .gate_supervision import (
    extract_theta,
    compute_utilization,
    utilization_to_target,
    apply_birth_prior,
)


class CFSDTMTrainer:
    """
    Two-phase training loop for CF-SDTM.

    Phase 1 (warmup): gates frozen at g≈1, only beta/theta trained.
    Phase 2 (fine-tune): gates initialised from data, trained jointly.

    Parameters
    ----------
    model                : CFSDTM instance
    dataset              : TopMost DynamicDataset — must expose .train_dataloader
    epochs               : DEPRECATED — use finetune_epochs; kept for backward compat
    lr                   : Adam learning rate                        (default 0.002)
    grad_clip            : max gradient norm; None = no clipping     (default 2.0)
    save_dir             : directory for periodic checkpoints; None = disabled
    save_every           : save checkpoint every N epochs            (default 200)
    log_every            : print progress every N epochs             (default 50)
    device               : 'cuda' or 'cpu'
    warmup_epochs        : epochs with frozen gates                  (default 80)
    finetune_epochs      : epochs of gate fine-tuning                (default 300)
    lambda_sparse        : weight for binary-entropy sparsity loss   (default 0.1)
    lambda_smooth        : weight for temporal smoothness loss       (default 0.05)
    lambda_sup           : weight for supervision MSE loss           (default 8.0)
    gate_temp            : temperature for z-normalisation sigmoid   (default 2.0)
    cov_threshold        : CoV below this → topic is stable          (default 0.25)
    stable_target        : gate target for stable (ever-present) topics (default 0.85)
    birth_rise_threshold : Δhalf threshold for birth detection       (default 0.25)
    birth_min_target     : minimum gate target after birth point     (default 0.65)
    warmup_sparse        : curriculum epochs for sparse loss ramp-up (default 30)
    """

    def __init__(
        self,
        model,
        dataset,
        epochs:               Optional[int]   = None,
        lr:                   float = 0.002,
        grad_clip:            Optional[float] = 2.0,
        save_dir:             Optional[str]   = None,
        save_every:           int   = 200,
        log_every:            int   = 50,
        device:               str   = 'cuda',
        warmup_epochs:        int   = 80,
        finetune_epochs:      int   = 300,
        lambda_sparse:        float = 0.1,
        lambda_smooth:        float = 0.05,
        lambda_sup:           float = 8.0,
        gate_temp:            float = 2.0,
        cov_threshold:        float = 0.25,
        stable_target:        float = 0.85,
        birth_rise_threshold: float = 0.25,
        birth_min_target:     float = 0.65,
        warmup_sparse:        int   = 30,
    ):
        if epochs is not None:
            warnings.warn(
                "The 'epochs' parameter is deprecated; use 'finetune_epochs' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            finetune_epochs = epochs

        self.model                = model.to(device)
        self.dataset              = dataset
        self.lr                   = lr
        self.device               = device
        self.grad_clip            = grad_clip
        self.save_dir             = save_dir
        self.save_every           = save_every
        self.log_every            = log_every
        self.warmup_epochs        = warmup_epochs
        self.finetune_epochs      = finetune_epochs
        self.lambda_sparse        = lambda_sparse
        self.lambda_smooth        = lambda_smooth
        self.lambda_sup           = lambda_sup
        self.gate_temp            = gate_temp
        self.cov_threshold        = cov_threshold
        self.stable_target        = stable_target
        self.birth_rise_threshold = birth_rise_threshold
        self.birth_min_target     = birth_min_target
        self.warmup_sparse        = warmup_sparse

        self._U_fixed      = None   # supervision target fixed after warmup
        self._utilization  = None   # raw utilization tensor from warmup
        self._cov          = None   # CoV per topic
        self._stable_mask  = None   # bool mask of stable topics
        self._birth_report = []

        self.history: dict = {
            'loss'         : [],   # ELBO per epoch (warmup + finetune)
            'gate_loss'    : [],   # sparse + smooth loss (finetune only)
            'sup_loss'     : [],   # supervision loss (finetune only)
            'active_ratio' : [],   # fraction of gates > threshold (finetune only)
            'cov'          : None, # np.ndarray (K,) set once after warmup
            'stable_topics': [],   # list[int] indices of stable topics
            'birth_report' : [],   # list[str] from apply_birth_prior
        }

    # ── Batch parsing ────────────────────────────────────────────────────────

    def _parse_batch(self, batch):
        """Extract (bow, time_ids) from DataLoader batch (dict or tuple)."""
        if isinstance(batch, dict):
            x = None
            for key in ('bow', 'x', 'data'):
                if batch.get(key) is not None:
                    x = batch[key]
                    break
            t = None
            for key in ('times', 'time_id', 'time_ids', 't'):
                if batch.get(key) is not None:
                    t = batch[key]
                    break
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
            t = batch[1] if len(batch) > 1 else None
        else:
            raise TypeError(f'Unexpected batch type: {type(batch)}')

        x = x.float().to(self.device)
        if t is not None:
            t = t.long().to(self.device)
        return x, t

    # ── Phase 1: Warmup ──────────────────────────────────────────────────────

    def _run_warmup(self) -> None:
        """Train beta/theta for warmup_epochs with gates frozen at g≈1."""
        model  = self.model
        loader = self.dataset.train_dataloader

        model.gate.logits.requires_grad_(False)
        model.gate.logits.data.fill_(10.0)  # sigmoid(10) ≈ 1.0

        params_no_gate = [p for n, p in model.named_parameters()
                          if 'gate' not in n]
        optimizer = torch.optim.Adam(params_no_gate, lr=self.lr)

        print(f'[CF-SDTM] Warmup: {self.warmup_epochs} epochs, gates frozen at g≈1.')

        for epoch in range(1, self.warmup_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                x, t = self._parse_batch(batch)
                optimizer.zero_grad()

                rst  = model.base(x, t)
                loss = rst['loss'] if isinstance(rst, dict) else rst[0]

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(params_no_gate, self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_loss)

            if epoch % self.log_every == 0:
                print(
                    f'  Warmup  {epoch:4d}/{self.warmup_epochs}  '
                    f'elbo {avg_loss:9.2f}'
                )

    # ── Gate initialisation ──────────────────────────────────────────────────

    def _init_gates(self) -> None:
        """Compute utilization from warmup model; initialise gate logits."""
        model = self.model

        print('[CF-SDTM] Computing topic utilization from warmup model…')
        utilization = compute_utilization(
            model,
            self.dataset.train_dataloader,
            self.dataset.num_times,
            model.num_topics,
            self.device,
        )
        self._utilization = utilization

        target, CoV, stable_mask = utilization_to_target(
            utilization,
            temp          = self.gate_temp,
            cov_threshold = self.cov_threshold,
            stable_target = self.stable_target,
        )
        self._cov         = CoV
        self._stable_mask = stable_mask

        target, birth_report = apply_birth_prior(
            target,
            rise_threshold = self.birth_rise_threshold,
            min_target     = self.birth_min_target,
        )
        self._birth_report = birth_report

        stable_indices = stable_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(stable_indices, int):
            stable_indices = [stable_indices]

        self.history['cov']           = CoV.cpu().numpy()
        self.history['stable_topics'] = stable_indices
        self.history['birth_report']  = birth_report

        if stable_indices:
            print(f'  Stable topics (CoV < {self.cov_threshold}): {stable_indices}')
        for line in birth_report:
            print(f'  Birth prior: {line}')

        model.gate.logits.requires_grad_(True)
        with torch.no_grad():
            tgt_clipped = target.clamp(0.05, 0.95)
            model.gate.logits.data = torch.log(
                tgt_clipped / (1.0 - tgt_clipped)
            ).to(self.device)

        # Supervision target is fixed — not updated during fine-tune
        self._U_fixed = target.clone().to(self.device)
        print('[CF-SDTM] Gate logits initialised from utilization targets.')

    # ── Phase 2: Fine-tune ───────────────────────────────────────────────────

    def _run_finetune(self) -> None:
        """Fine-tune all parameters with sparse + smooth + supervision losses."""
        model          = self.model
        loader         = self.dataset.train_dataloader
        U_fixed        = self._U_fixed
        gate_threshold = getattr(model, 'gate_threshold', 0.3)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr * 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=150, gamma=0.5
        )

        print(
            f'[CF-SDTM] Fine-tune: {self.finetune_epochs} epochs  '
            f'lambda_sup={self.lambda_sup}  warmup_sparse={self.warmup_sparse}'
        )

        for epoch in range(1, self.finetune_epochs + 1):
            model.train()
            epoch_elbo  = 0.0
            epoch_gloss = 0.0
            epoch_sup   = 0.0
            n_batches   = 0

            sparse_weight = self.lambda_sparse * min(
                1.0, epoch / max(self.warmup_sparse, 1)
            )

            for batch in loader:
                x, t = self._parse_batch(batch)
                optimizer.zero_grad()

                rst  = model.base(x, t)
                elbo = rst['loss'] if isinstance(rst, dict) else rst[0]

                if not torch.isfinite(elbo):
                    continue

                g      = torch.sigmoid(model.gate.logits)           # (T, K)
                sparse = (g * (1 - g)).mean()
                smooth = ((g[1:] - g[:-1]) ** 2).mean() \
                         if g.shape[0] > 1 else g.new_tensor(0.0)
                sup    = F.mse_loss(g, U_fixed.detach())

                total = (elbo
                         + sparse_weight * sparse
                         + self.lambda_smooth * smooth
                         + self.lambda_sup * sup)

                total.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                # Gate floor: keeps inactive gates binary (sigmoid(-4) ≈ 0.018)
                with torch.no_grad():
                    model.gate.logits.data.clamp_(min=-4.0)

                epoch_elbo  += elbo.item()
                epoch_gloss += (sparse_weight * sparse
                                + self.lambda_smooth * smooth).item()
                epoch_sup   += (self.lambda_sup * sup).item()
                n_batches   += 1

            if n_batches == 0:
                scheduler.step()
                continue

            avg_elbo  = epoch_elbo  / n_batches
            avg_gloss = epoch_gloss / n_batches
            avg_sup   = epoch_sup   / n_batches

            with torch.no_grad():
                gates  = torch.sigmoid(model.gate.logits)
                active = (gates > gate_threshold).float().mean().item()

            self.history['loss'].append(avg_elbo)
            self.history['gate_loss'].append(avg_gloss)
            self.history['sup_loss'].append(avg_sup)
            self.history['active_ratio'].append(active)

            scheduler.step()

            if epoch % self.log_every == 0:
                print(
                    f'  Fine-tune {epoch:4d}/{self.finetune_epochs}  '
                    f'elbo {avg_elbo:9.2f}  '
                    f'gate_loss {avg_gloss:.4f}  '
                    f'sup_loss {avg_sup:.4f}  '
                    f'active {active:.1%}'
                )

            if self.save_dir and epoch % self.save_every == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(
                    self.save_dir,
                    f'cfsdtm_finetune_epoch{epoch:04d}.pt',
                )
                model.save_checkpoint(path)

    # ── Main entry point ─────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Run two-phase training: warmup then fine-tune.

        Returns
        -------
        history : dict with keys:
          'loss'          — ELBO per epoch (warmup + finetune concatenated)
          'gate_loss'     — sparse + smooth loss per finetune epoch
          'sup_loss'      — supervision loss per finetune epoch
          'active_ratio'  — fraction of gates > threshold per finetune epoch
          'cov'           — np.ndarray (K,) CoV per topic, set after warmup
          'stable_topics' — list[int] indices of stable topics
          'birth_report'  — list[str] from apply_birth_prior
        """
        self._run_warmup()
        self._init_gates()
        self._run_finetune()
        return self.history

    # ── Analysis helpers ─────────────────────────────────────────────────────

    def get_gate_summary(self) -> dict:
        """
        Return a summary of final gate state and training diagnostics.

        Returns
        -------
        dict with keys:
          'gates'         — np.ndarray (T, K) final gate values sigmoid(logits)
          'active_mask'   — bool ndarray (T, K) gates > gate_threshold
          'cov'           — np.ndarray (K,) coefficient of variation per topic
          'stable_topics' — list[int] indices of stable (ever-present) topics
          'utilization'   — np.ndarray (T, K) raw utilization from warmup
        """
        model          = self.model
        gate_threshold = getattr(model, 'gate_threshold', 0.3)

        with torch.no_grad():
            gates = torch.sigmoid(model.gate.logits).cpu().numpy()

        cov  = self._cov.cpu().numpy()         if self._cov         is not None else None
        util = self._utilization.cpu().numpy() if self._utilization is not None else None

        return {
            'gates'        : gates,
            'active_mask'  : gates > gate_threshold,
            'cov'          : cov,
            'stable_topics': self.history.get('stable_topics', []),
            'utilization'  : util,
        }

    def get_train_theta(self) -> np.ndarray:
        """
        Collect document-topic distributions for the full training set.

        Uses extract_theta in model.train() mode to avoid the CFDTM eval-mode
        bug (different return value signature in train vs eval).

        Returns
        -------
        np.ndarray (N_train, K)
        """
        model  = self.model
        chunks = []

        model.train()
        with torch.no_grad():
            for batch in self.dataset.train_dataloader:
                x, _ = self._parse_batch(batch)
                theta = extract_theta(model, x, self.device)
                chunks.append(theta.cpu().numpy())

        return np.vstack(chunks)

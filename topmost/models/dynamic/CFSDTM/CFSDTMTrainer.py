"""
CFSDTMTrainer.py — Custom training loop for CF-SDTM
====================================================
Handles the batch formats that TopMost DynamicDataset DataLoaders produce
(dict with 'bow'/'times' keys, or plain tuple).

Key difference from DynamicTrainer
------------------------------------
- Clips gradients (recommended when gate loss is added)
- Logs active-topic ratio every `log_every` epochs so you can watch birth/death
- Saves checkpoints via model.save_checkpoint() (custom format, not state_dict)
"""

from __future__ import annotations

import os
import time
import torch
import numpy as np
from typing import Optional


class CFSDTMTrainer:
    """
    Training loop for CF-SDTM.

    Parameters
    ----------
    model       : CFSDTM instance
    dataset     : TopMost DynamicDataset — must expose .train_dataloader
    epochs      : number of training epochs              (default 800)
    lr          : Adam learning rate                     (default 0.002)
    grad_clip   : max gradient norm; None = no clipping  (default 2.0)
    save_dir    : directory for periodic checkpoints; None = no saving
    save_every  : save a checkpoint every N epochs       (default 200)
    log_every   : print progress every N epochs          (default 50)
    device      : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model,
        dataset,
        epochs:    int   = 800,
        lr:        float = 0.002,
        grad_clip: Optional[float] = 2.0,
        save_dir:  Optional[str]   = None,
        save_every: int  = 200,
        log_every:  int  = 50,
        device:    str   = 'cuda',
    ):
        self.model     = model.to(device)
        self.dataset   = dataset
        self.epochs    = epochs
        self.device    = device
        self.grad_clip = grad_clip
        self.save_dir  = save_dir
        self.save_every = save_every
        self.log_every  = log_every

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.history: dict[str, list] = {
            'loss'        : [],
            'gate_loss'   : [],
            'active_ratio': [],
        }

    # ── Batch parsing ────────────────────────────────────────────────────────

    def _parse_batch(self, batch):
        """
        Extract (bow, time_ids) from whatever format the DataLoader returns.

        Uses explicit None-checks to avoid mishandling all-zero BoW tensors.
        """
        if isinstance(batch, dict):
            x = None
            for key in ['bow', 'x', 'data']:
                if batch.get(key) is not None:
                    x = batch[key]
                    break

            t = None
            for key in ['times', 'time_id', 'time_ids', 't']:
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

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Run training for self.epochs epochs.

        Returns
        -------
        history : dict with keys 'loss', 'gate_loss', 'active_ratio'
        """
        loader = self.dataset.train_dataloader

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.time()

            epoch_loss  = 0.0
            epoch_gloss = 0.0
            n_batches   = 0

            for batch in loader:
                x, t = self._parse_batch(batch)
                self.optimizer.zero_grad()

                out = self.model(x, t)
                if isinstance(out, dict):
                    loss = out['loss']
                elif isinstance(out, (tuple, list)):
                    loss = out[0]
                else:
                    loss = out

                if not torch.isfinite(loss):
                    continue  # skip corrupt batch; do not accumulate

                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

                epoch_loss  += loss.item()
                epoch_gloss += self.model.gate_loss().item()
                n_batches   += 1

            if n_batches == 0:
                continue

            avg_loss  = epoch_loss  / n_batches
            avg_gloss = epoch_gloss / n_batches

            with torch.no_grad():
                gates  = self.model.get_gates()
                active = (gates > self.model.gate_threshold).float().mean().item()

            self.history['loss'].append(avg_loss)
            self.history['gate_loss'].append(avg_gloss)
            self.history['active_ratio'].append(active)

            if epoch % self.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f'Epoch {epoch:4d}/{self.epochs}  '
                    f'loss {avg_loss:9.2f}  '
                    f'gate_loss {avg_gloss:.4f}  '
                    f'active {active:.1%}  '
                    f'({elapsed:.1f}s)'
                )

            if self.save_dir and epoch % self.save_every == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(
                    self.save_dir, f'cfsdtm_epoch{epoch:04d}.pt'
                )
                self.model.save_checkpoint(path)

        return self.history

    # ── Inference ────────────────────────────────────────────────────────────

    def get_train_theta(self) -> np.ndarray:
        """
        Collect gated document-topic distributions for the full training set.

        Returns numpy array of shape (N_train, K).
        """
        self.model.eval()
        chunks = []

        with torch.no_grad():
            for batch in self.dataset.train_dataloader:
                x, t  = self._parse_batch(batch)
                theta = self.model.get_theta(x, t)
                chunks.append(theta.cpu().numpy())

        return np.vstack(chunks)

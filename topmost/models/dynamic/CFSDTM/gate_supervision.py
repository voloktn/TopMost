"""
gate_supervision.py — Data-driven supervision for CF-SDTM topic-activity gates.

These helpers implement the warmup-based gate initialisation that solves the
gate collapse / gate stagnation problem: without a data signal, gates trained
only on sparsity + smoothness losses either freeze at g=0.5 or collapse to 0.

Workflow called by CFSDTMTrainer after the warmup phase:
  1. compute_utilization  — measure which topics are used in which time slices
  2. utilization_to_target — z-normalise to get gate targets in (0, 1)
  3. apply_birth_prior    — prevent gates closing for freshly-born topics
"""

import torch
import torch.nn.functional as F


def extract_theta(model, bow, device='cpu'):
    """
    Reliably extract document-topic distribution theta from the model.

    Must be called while model is in train() mode.  CFDTM.get_theta returns
    a different number of values in eval mode (known bug in the base model).
    Should be called inside torch.no_grad() to avoid accumulating gradients.

    Attempt order:
      1. model.get_theta(bow)      — if tuple/list, take [0]
      2. model.base.get_theta(bow) — if tuple/list, take [0]
      3. model.encode(bow)         — take first element, apply softmax
      4. RuntimeError listing child module names for diagnosis

    Parameters
    ----------
    model  : CFSDTM instance (must be in train() mode)
    bow    : bag-of-words tensor (B, V), float
    device : target device string

    Returns
    -------
    theta : torch.Tensor (B, K), document-topic distribution
    """
    bow = bow.to(device)

    # Attempt 1: model.get_theta
    try:
        result = model.get_theta(bow)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result
    except Exception:
        pass

    # Attempt 2: model.base.get_theta  (train mode → returns (theta, mu, logvar))
    try:
        result = model.base.get_theta(bow)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result
    except Exception:
        pass

    # Attempt 3: model.encode
    try:
        result = model.encode(bow)
        first = result[0] if isinstance(result, (tuple, list)) else result
        return F.softmax(first, dim=-1)
    except Exception:
        pass

    child_names = [name for name, _ in model.named_children()]
    raise RuntimeError(
        f"Cannot extract theta from model. Child modules: {child_names}"
    )


def compute_utilization(model, dataloader, num_times, num_topics, device='cpu'):
    """
    Compute topic utilization matrix U[t, k] = mean theta[k] over all
    documents in time slice t.

    Parameters
    ----------
    model       : CFSDTM instance
    dataloader  : DataLoader yielding dicts with 'bow' and 'times' keys
    num_times   : number of time slices T
    num_topics  : number of topics K
    device      : target device string

    Returns
    -------
    utilization : torch.Tensor (T, K) — mean document-topic weight per slice
    """
    model.train()

    sums   = torch.zeros(num_times, num_topics, device=device)
    counts = torch.zeros(num_times, 1,          device=device)

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                bow = batch.get('bow')
                if bow is None: bow = batch.get('x')
                if bow is None: bow = batch.get('data')
                times = batch.get('times')
                if times is None: times = batch.get('time_id')
                if times is None: times = batch.get('time_ids')
                if times is None: times = batch.get('t')
            elif isinstance(batch, (list, tuple)):
                bow   = batch[0]
                times = batch[1] if len(batch) > 1 else None
            else:
                raise TypeError(f'Unexpected batch type: {type(batch)}')

            bow   = bow.float().to(device)
            times = times.long().to(device)

            theta = extract_theta(model, bow, device)  # (B, K)

            for t in range(num_times):
                mask = (times == t)
                if mask.any():
                    sums[t]   += theta[mask].sum(dim=0)
                    counts[t] += mask.sum()

    utilization = sums / counts.clamp(min=1)
    return utilization


def utilization_to_target(utilization, temp=2.0, cov_threshold=0.25,
                           stable_target=0.85):
    """
    Convert raw utilization matrix to gate targets via z-normalisation with
    CoV-based detection of stable (ever-present) topics.

    Why z-normalisation instead of raw utilization:
      VAE encoders spread probability mass evenly — even a dominant topic gets
      ~0.2 utilization, not 0.8.  Using raw values as targets would pull all
      gates toward zero.  Z-normalisation captures relative activity across
      time slices rather than absolute values.

    Algorithm
    ---------
    Step 1 — Z-normalise per topic, then sigmoid:
      U_mean = utilization.mean(dim=0)          # (1, K)
      U_std  = utilization.std(dim=0)           # (1, K), clamped ≥ 1e-6
      U_z    = (utilization - U_mean) / U_std   # (T, K)
      target = sigmoid(U_z * temp)              # (T, K) ∈ (0, 1)

    Step 2 — CoV detector for stable (ever-present) topics:
      CoV = U_std / U_mean                      # (K,)
      Stable topics have low temporal variation → z-score ≈ 0 everywhere
      → sigmoid(0) = 0.5 → supervision is uninformative.
      Fix: force target = stable_target for all slices of stable topics.

    Parameters
    ----------
    utilization   : torch.Tensor (T, K) — output of compute_utilization
    temp          : float — temperature scaling for sigmoid       (default 2.0)
    cov_threshold : float — CoV below this → topic is stable     (default 0.25)
    stable_target : float — gate target for stable topics        (default 0.85)

    Returns
    -------
    target      : torch.Tensor (T, K) gate targets ∈ (0, 1)
    CoV         : torch.Tensor (K,) coefficient of variation per topic
    stable_mask : torch.BoolTensor (K,) True = stable (ever-present) topic
    """
    U_mean = utilization.mean(dim=0, keepdim=True)                    # (1, K)
    U_std  = utilization.std(dim=0, keepdim=True).clamp(min=1e-6)     # (1, K)
    U_z    = (utilization - U_mean) / U_std                            # (T, K)
    target = torch.sigmoid(U_z * temp)                                 # (T, K)

    CoV         = (U_std / U_mean.clamp(min=1e-6)).squeeze(0)         # (K,)
    stable_mask = CoV < cov_threshold                                  # (K,)

    target[:, stable_mask] = stable_target

    return target, CoV, stable_mask


def apply_birth_prior(target, rise_threshold=0.25, min_target=0.65):
    """
    Apply a monotonicity prior for topics that are being born (rising utilization).

    Problem: a rising topic can receive a low z-score in its later time slices
    if a stronger topic competes there, causing the gate to close even though
    the topic is already active and should stay open.

    Algorithm
    ---------
    For each topic k:
      first_half  = mean(target[:T//2, k])
      second_half = mean(target[T//2:, k])
      Δhalf = second_half - first_half

      If Δhalf > rise_threshold (clearly rising topic):
        t_start = first slice t where target[t, k] > 0.5
        For all t >= t_start: target[t, k] = max(target[t, k], min_target)

    Parameters
    ----------
    target          : torch.Tensor (T, K) — output of utilization_to_target
    rise_threshold  : float — Δhalf threshold for birth detection  (default 0.25)
    min_target      : float — minimum target value after birth     (default 0.65)

    Returns
    -------
    target  : torch.Tensor (T, K) with birth prior applied (returns a copy)
    report  : list[str] description of corrections applied per topic
    """
    target = target.clone()
    T_len  = target.shape[0]
    K      = target.shape[1]
    report = []

    for k in range(K):
        first_half  = target[:T_len // 2, k].mean().item()
        second_half = target[T_len // 2:, k].mean().item()
        delta_half  = second_half - first_half

        if delta_half > rise_threshold:
            above = (target[:, k] > 0.5).nonzero(as_tuple=False)
            if len(above) > 0:
                t_start = above[0].item()
                old_min = target[t_start:, k].min().item()
                floor   = target.new_full(target[t_start:, k].shape, min_target)
                target[t_start:, k] = torch.max(target[t_start:, k], floor)
                report.append(
                    f'Topic {k}: birth prior applied from t={t_start} '
                    f'(Δhalf={delta_half:.3f}, '
                    f'old_min={old_min:.3f} → {min_target})'
                )

    return target, report

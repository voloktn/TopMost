from .CFSDTM import CFSDTM, TopicGate
from .CFSDTMTrainer import CFSDTMTrainer
from .gate_supervision import (
    extract_theta,
    compute_utilization,
    utilization_to_target,
    apply_birth_prior,
)

__all__ = [
    'CFSDTM', 'TopicGate', 'CFSDTMTrainer',
    'extract_theta', 'compute_utilization',
    'utilization_to_target', 'apply_birth_prior',
]

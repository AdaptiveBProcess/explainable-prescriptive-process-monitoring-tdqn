from __future__ import annotations

import torch


def apply_action_mask(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply action mask to Q-values (invalid actions get large negative value).

    q_values: (batch, n_actions)
    mask: (batch, n_actions) with 1 for valid, 0 for invalid.
    """
    invalid = mask < 0.5
    q_values = q_values.clone()
    q_values[invalid] = -1e9
    return q_values



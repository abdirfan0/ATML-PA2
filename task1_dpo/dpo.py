from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    preference_margin = policy_margin - ref_margin
    logits = beta * preference_margin
    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "preference_accuracy": (
            preference_margin > 0
        ).float().mean().detach(),
    }

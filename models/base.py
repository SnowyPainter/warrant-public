from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelResult:
    logits: torch.Tensor
    aux: dict[str, torch.Tensor]


class WarrantModel(nn.Module):
    model_name = "warrant_model"

    def __init__(self, *, use_warrant: bool = False) -> None:
        super().__init__()
        self.use_warrant = bool(use_warrant)

    @staticmethod
    def aux_from_info(info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gate = info.get("warrant_gate")
        logits = info.get("warrant_logits")
        aux: dict[str, torch.Tensor] = {}
        if gate is not None:
            aux["warrant_gate_mean"] = gate.detach().mean()
        if logits is not None:
            aux["warrant_logit_mean"] = logits.detach().mean()
        for key in ("edge_warrant_gate_mean", "edge_warrant_attention_entropy"):
            value = info.get(key)
            if value is not None:
                aux[key] = value.detach()
        return aux

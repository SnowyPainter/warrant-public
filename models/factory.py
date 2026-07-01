from __future__ import annotations

from typing import Any

from .ctdg import DyGFormer, GraphMixer, TGAT
from .mtpp import AttNHP, SAHP, THP
from .official import build_official_model
from .rag import FiD, LED
from .stpp import DeepSTPP, NSTPP, TransformerSTPP
from .tkg import CyGNet, RENET, XERTE
from .warrant_patches import replace_torch_multihead_attention


MODEL_REGISTRY = {
    "TGAT": TGAT,
    "DyGFormer": DyGFormer,
    "GraphMixer": GraphMixer,
    "SAHP": SAHP,
    "THP": THP,
    "AttNHP": AttNHP,
    "Transformer-STPP": TransformerSTPP,
    "NSTPP": NSTPP,
    "DeepSTPP": DeepSTPP,
    "RE-NET": RENET,
    "xERTE": XERTE,
    "CyGNet": CyGNet,
    "FiD": FiD,
    "LED": LED,
}


def build_model(name: str, *, use_warrant: bool = False, implementation: str = "reference", **kwargs: Any):
    if implementation == "official":
        return build_official_model(name, use_warrant=use_warrant, **kwargs)
    if implementation not in {"reference", "native", "lightweight"}:
        raise ValueError("implementation must be one of: official, reference, native, lightweight")
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")
    model = MODEL_REGISTRY[name](use_warrant=use_warrant, **kwargs)
    replacements = 0
    if use_warrant and name not in {"FiD", "LED"}:
        replacements = replace_torch_multihead_attention(
            model,
            gate_init=float(kwargs.get("gate_init", 0.95)),
            gate_leak=float(kwargs.get("gate_leak", 0.05)),
        )
    model.warrant_replacements = replacements
    return model


__all__ = ["MODEL_REGISTRY", "build_model"]

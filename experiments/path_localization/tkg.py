from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.neural_dissection.run import TKGRunner, batches, finite_mean, gate_stats
from experiments.path_localization.common import decompose_warrant_block, set_attention_warrant
from models.attention import WarrantedAttention
from experiments.path_localization.rag import _nontrivial_permutation


class PathTKGRunner(TKGRunner):
    """TKG tail-prediction path-localization runner.

    The shuffled control keeps historical facts and the target tail fixed but
    swaps the current (head, relation, time) query across the batch.
    """

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.spec.variant != "shuffled_pairing":
            return super().evaluate(model)

        model.eval()
        losses, correct, total, rr, aux_rows = [], 0, 0, [], []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                *inputs, target = self.make_batch(rows)
                if target.shape[0] > 1:
                    order = _nontrivial_permutation(target.shape[0], target.device)
                    inputs[0] = inputs[0][order]
                    inputs[1] = inputs[1][order]
                    inputs[2] = inputs[2][order]
                result = model(*inputs)
                losses.append(float(F.cross_entropy(result.logits, target).cpu()))
                correct += int((result.logits.argmax(dim=-1) == target).sum().cpu())
                total += int(target.numel())
                target_logits = result.logits.gather(1, target.unsqueeze(1))
                ranks = (result.logits > target_logits).sum(dim=1).float() + 1.0
                rr.extend((1.0 / ranks).cpu().tolist())
                aux_rows.append(result.aux)
        mrr = finite_mean(rr)
        return {"eval_loss": finite_mean(losses), "primary_metric": mrr, "mrr": mrr, "accuracy": correct / max(1, total), **gate_stats(aux_rows)}


def configure_model(model: torch.nn.Module, variant: str) -> None:
    if variant == "open_path_no_gate":
        set_attention_warrant(model, False)
        model.tail_path_mode = "open_no_gate"
        model.warrant_expected = False
        return
    if variant in {"correct_path_warrant", "shuffled_pairing", "combined_full"}:
        set_attention_warrant(model, True)
        model.tail_path_mode = "warrant"
        return
    if variant in {"scalar_gate", "item_only_gate", "normalized_gate", "attention_adapter"}:
        set_attention_warrant(model, True)
        model.tail_path_mode = "warrant"
        for module in model.modules():
            if isinstance(module, WarrantedAttention):
                decompose_warrant_block(module.warrant, variant)
        return
    if variant == "generic_open_path":
        set_attention_warrant(model, True)
        model.tail_path_mode = "open_no_gate"

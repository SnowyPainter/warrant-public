from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.neural_dissection.run import MTPPRunner, batches, finite_mean, gate_stats
from experiments.path_localization.common import decompose_warrant_block, force_open_warrant_block
from experiments.path_localization.rag import _nontrivial_permutation


class PathMTPPRunner(MTPPRunner):
    """MTPP next-mark path-localization runner.

    The shuffled control keeps target marks fixed but swaps the historical
    event sequence that defines the candidate-wise prediction query.
    """

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.spec.variant != "shuffled_pairing":
            return super().evaluate(model)

        model.eval()
        losses, correct, total, rr, maes, aux_rows = [], 0, 0, [], [], []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                marks, times, targets, dtime, mask = self.make_batch(rows)
                if targets.shape[0] > 1:
                    order = _nontrivial_permutation(targets.shape[0], targets.device)
                    marks = marks[order]
                    times = times[order]
                    mask = mask[order]
                result = model(marks, times, mask)
                loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
                target_logits = result.logits.gather(1, targets.unsqueeze(1))
                ranks = (result.logits > target_logits).sum(dim=1).float() + 1.0
                rr.extend((1.0 / ranks).cpu().tolist())
                correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                total += int(targets.numel())
                maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
                losses.append(float(loss.cpu()))
                aux_rows.append(result.aux)
        mark_mrr = finite_mean(rr)
        return {
            "eval_loss": finite_mean(losses),
            "primary_metric": mark_mrr,
            "mark_mrr": mark_mrr,
            "accuracy": correct / max(1, total),
            "mae_time": finite_mean(maes),
            **gate_stats(aux_rows),
        }


def configure_model(model: torch.nn.Module, variant: str) -> None:
    if variant in {"open_path_no_gate", "generic_open_path"}:
        if getattr(model, "candidate_warrant", None) is None:
            raise ValueError("MTPP open_path_no_gate requires a candidate Warrant path")
        force_open_warrant_block(model.candidate_warrant)
    elif variant in {"scalar_gate", "item_only_gate", "normalized_gate", "attention_adapter"}:
        if getattr(model, "candidate_warrant", None) is None:
            raise ValueError(f"MTPP {variant} requires a candidate Warrant path")
        decompose_warrant_block(model.candidate_warrant, variant)
    return None

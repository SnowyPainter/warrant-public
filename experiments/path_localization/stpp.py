from __future__ import annotations

import torch

from evaluation.evaluate import stpp_joint_nll
from experiments.neural_dissection.run import STPPRunner, batches, finite_mean, gate_stats
from experiments.path_localization.common import force_open_warrant_block
from experiments.path_localization.rag import _nontrivial_permutation


class PathSTPPRunner(STPPRunner):
    """STPP next-location path-localization runner.

    The shuffled control keeps the next-location target fixed but swaps the
    spatio-temporal history used to build the prediction dynamics.
    """

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.spec.variant != "shuffled_pairing":
            return super().evaluate(model)

        model.eval()
        losses, correct, total, rmses, maes, aux_rows = [], 0, 0, [], [], []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                times, coords, marks, targets, target_coords, dtime, mask = self.make_batch(rows)
                if targets.shape[0] > 1:
                    order = _nontrivial_permutation(targets.shape[0], targets.device)
                    times = times[order]
                    coords = coords[order]
                    marks = marks[order]
                    mask = mask[order]
                result = model(times, coords, marks, mask)
                loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=self.num_marks)
                losses.append(float(loss.cpu()))
                if self.num_marks > 1:
                    correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                    total += int(targets.numel())
                rmses.extend(torch.sqrt(((result.aux["next_location"] - target_coords) ** 2).sum(dim=-1)).cpu().tolist())
                maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
                aux_rows.append(result.aux)
        joint_nll = finite_mean(losses)
        rmse_location = finite_mean(rmses)
        return {
            "eval_loss": joint_nll,
            "primary_metric": rmse_location,
            "joint_nll": joint_nll,
            "accuracy": correct / max(1, total) if total else float("nan"),
            "rmse_location": rmse_location,
            "mae_time": finite_mean(maes),
            **gate_stats(aux_rows),
        }


def configure_model(model: torch.nn.Module, variant: str) -> None:
    if variant == "open_path_no_gate":
        if getattr(model, "prediction_warrant", None) is None:
            raise ValueError("STPP open_path_no_gate requires a prediction Warrant path")
        force_open_warrant_block(model.prediction_warrant)
    return None

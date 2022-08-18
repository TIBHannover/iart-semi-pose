from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat

import numpy as np
from sklearn.metrics import fbeta_score
import logging


class FBetaMetric(Metric):

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: int,
        compute_on_step: bool = False,
        thresholds: List[float] = [0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
        beta: float = 2,
        mask: List[float] = None,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.thresholds = thresholds
        self.beta = beta
        self.mask = mask
        if mask is None:
            self.mask = torch.ones(num_classes, dtype=torch.bool)

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `FBeta` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        self.preds.append(preds)
        self.targets.append(target)

    def binarize_prediction(self, probabilities, threshold: float, argsorted=None, min_labels=1, max_labels=10):
        """Return matrix of 0/1 predictions, same shape as probabilities."""
        # assert probabilities.shape[1] == self.num_classes
        if argsorted is None:
            argsorted = probabilities.argsort(axis=1)

        max_mask = self._make_mask(argsorted, max_labels)
        min_mask = self._make_mask(argsorted, min_labels)
        prob_mask = probabilities > threshold

        return (max_mask & prob_mask) | min_mask

    def _make_mask(self, argsrtd, top_n: int):
        mask = np.zeros_like(argsrtd, dtype=np.uint8)
        col_indices = argsrtd[:, -top_n:].reshape(-1)
        row_indices = [i // top_n for i in range(len(col_indices))]

        mask[row_indices, col_indices] = 1
        return mask

    def get_score(self, y_pred, all_targets):
        mask = np.sum(all_targets, axis=0)
        score = fbeta_score(all_targets, y_pred, beta=2, average=None, zero_division=0.0)
        score[mask == 0] = np.nan
        return score

    def compute(self) -> Union[Tensor, List[Tensor]]:
        """Compute the average precision score.
        Returns:
            tensor with average precision. If multiclass will return list
            of such tensors, one for each class
        """
        # logging.info(len(self.preds))
        preds = dim_zero_cat(self.preds).cpu().numpy()
        targets = dim_zero_cat(self.targets).cpu().numpy()

        # nonfiltered_lbs = np.where(~self.mask.numpy())
        # preds = np.delete(preds, nonfiltered_lbs, axis=1)
        # targets = np.delete(targets, nonfiltered_lbs, axis=1)
        metrics = {}
        for threshold in self.thresholds:
            try:
                metrics[f"{threshold:.2f}"] = torch.tensor(self.get_score((preds > threshold).astype(float), targets))
            except:
                metrics[f"{threshold:.2f}"] = np.zeros_like(preds)

        return metrics

    def mean(self, score_per_class, mask=None):

        nan_classes = torch.isnan(score_per_class)

        score_per_class[nan_classes] = 0.0
        if mask is not None:
            mask = (1 - nan_classes.float()) * mask
        else:
            mask = 1 - nan_classes.float()

        num_classes = torch.sum(mask)
      
        return (torch.sum(score_per_class * mask) / num_classes).float()

    @property
    def is_differentiable(self) -> bool:
        return False
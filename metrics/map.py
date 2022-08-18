from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat

import numpy as np
from sklearn.metrics import average_precision_score
import logging


class MAPMetric(Metric):

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: int,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `MAP` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Union[Tensor, List[Tensor]]:
        """Compute the average precision score.
        Returns:
            tensor with average precision. If multiclass will return list
            of such tensors, one for each class
        """
        # logging.info(len(self.preds))
        preds = dim_zero_cat(self.preds).cpu().numpy()
        targets = dim_zero_cat(self.targets).cpu().numpy()
        try:
            score_per_class = average_precision_score(targets, preds, average=None)
        except:
            score_per_class = np.zeros_like(preds)

        return torch.tensor(score_per_class)

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
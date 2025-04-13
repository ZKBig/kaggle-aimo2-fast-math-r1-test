import torch
import torch.nn as nn
from typing import Any, Optional, Union
from trl import GKDTrainer


class CustomGKDTrainer(GKDTrainer):
    '''
    A GKDTrainer without .generate_on_policy_outputs method
    '''
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        loss = super(GKDTrainer, self).training_step(model, inputs, num_items_in_batch)
        return loss

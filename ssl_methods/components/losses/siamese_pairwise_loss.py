from abc import ABC, abstractmethod
from typing import List

from torch import nn, Tensor


class SiamesePairwiseLoss(nn.Module, ABC):
    def forward(self, branches_outputs: List[List[Tensor]]) -> Tensor:
        branch_outputs = branches_outputs[0]

        loss = 0
        pair_count = 0

        for i, out1 in enumerate(branch_outputs):
            for j, out2 in enumerate(branch_outputs):
                if i != j:
                    loss += self._pair_loss(out1, out2)
                    pair_count += 1
        loss = loss / pair_count

        return loss

    @abstractmethod
    def _pair_loss(self, out1: Tensor, out2: Tensor) -> Tensor:
        pass

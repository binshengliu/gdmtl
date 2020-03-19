"""The rank hinge loss. Originated from matchzoo-py."""
import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class RankHingeLoss(nn.Module):  # type:ignore
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """

    __constants__ = ["num_neg", "margin", "reduction"]

    def __init__(
        self, num_neg: int = 1, margin: float = 1.0, reduction: str = "mean",
    ):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        :param neg_mode: String. Specifies how to handle multiple negatives.
            ``'avg'``: Compare averaged negatives against one positive.
            ``'ind'``: Compare individual negatives against repeated positives.
        """
        super().__init__()
        self.num_neg = num_neg
        self.margin = margin
        self.reduction = reduction

    def forward(  # type: ignore
        self, y_pred: torch.Tensor, y_true: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate rank hinge loss. Different from the matchzoo original
        version in that negative scores are not averaged but
        calculated individually.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.

        """
        shape = y_pred.size()
        num_groups = y_pred.size(0) // (self.num_neg + 1)
        y_pos = y_pred.view(num_groups, (self.num_neg + 1), *shape[1:])[:, :1, ...]
        y_pos = y_pos.expand(num_groups, self.num_neg, *shape[1:])
        y_neg = y_pred.view(num_groups, (self.num_neg + 1), *shape[1:])[:, 1:, ...]

        y_ones = torch.ones_like(y_pos)
        return F.margin_ranking_loss(
            y_pos, y_neg, y_ones, margin=self.margin, reduction=self.reduction
        )

    @property
    def num_neg(self) -> int:
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value: int) -> None:
        """`num_neg` setter."""
        self._num_neg = value

    @property
    def margin(self) -> float:
        """`margin` getter."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """`margin` setter."""
        self._margin = value


class UnlikelihoodLoss(nn.Module):  # type:ignore
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """

    __constants__ = ["num_neg", "margin", "reduction"]

    def __init__(self, num_neg: int = 1, unlikelihood_weight: float = 0.0):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.num_neg = num_neg
        self.unlikelihood_weight = unlikelihood_weight

    def forward(  # type: ignore
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate rank hinge loss. Different from the matchzoo original
        version in that negative scores are not averaged but
        calculated individually.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.

        """
        assert y_pred.dim() == 3
        assert y_true.dim() == 2

        bsz = y_pred.size(0) // (self.num_neg + 1)
        gsz = self.num_neg + 1
        seq_len, vocab_size = y_pred.size()[-2:]

        # Use F.log_softmax for efficiency and numerical stability.
        lprobs = F.log_softmax(y_pred, dim=-1).view(bsz, gsz, seq_len, vocab_size)
        y_true = y_true.view(bsz, gsz, seq_len)

        likely_lprobs = lprobs[:, :1].reshape(-1, vocab_size)
        likely_labels = y_true[:, :1].reshape(-1)
        likelihood_loss = F.nll_loss(likely_lprobs, likely_labels)
        if self.unlikelihood_weight <= 0:
            return likelihood_loss, likelihood_loss, likelihood_loss.new_tensor(0)

        unlikely_lprobs = lprobs[:, 1:].reshape(-1, vocab_size)
        unlikely_lprobs = torch.clamp(-unlikely_lprobs.exp() + 1, min=1e-5).log()
        unlikely_labels = y_true[:, 1:].reshape(-1)

        unlikelihood_loss = F.nll_loss(unlikely_lprobs, unlikely_labels)

        loss = likelihood_loss + self.unlikelihood_weight * unlikelihood_loss
        return loss, likelihood_loss, unlikelihood_loss


def uncertainty(loss: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
    return (loss / (sigma_sq * 2)) + (sigma_sq + 1).log()


class UncertaintyLoss(nn.Module):  # type: ignore
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = UncertaintyLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, min: Optional[float] = None) -> None:
        super(UncertaintyLoss, self).__init__()
        self._params = nn.Parameter(torch.ones(3))  # type: ignore
        self._prev = self._params.detach().clone()
        self._indexes = {"sum": 0, "qgen": 0, "qa": 1, "ranker": 2}
        self._min = min

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:  # type: ignore
        return uncertainty(x, self.__getitem__(mode, False))

    @torch.no_grad()
    def on_before_zero_grad(self, optimizer: Any) -> None:
        if self._min is not None and torch.any(self._params <= self._min):
            logger.debug(
                f"Restore uncertainty from {self._params.data} to {self._prev}."
            )
            self._params.copy_(self._prev)  # type: ignore
        self._prev.copy_(self._params)  # type: ignore

    def __getitem__(self, mode: str, clone: bool = True) -> torch.Tensor:
        val = self._params[self._indexes[mode]]
        if clone:
            val = val.detach().clone()
        return val

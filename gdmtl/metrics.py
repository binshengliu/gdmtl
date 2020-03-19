"""Mean reciprocal ranking metric. Originated from matchzoo-py."""
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


class Metric(ABC):
    ALIAS = [""]

    @abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        ...


class MeanReciprocalRank(Metric):
    """Mean reciprocal rank metric."""

    ALIAS = ["mean_reciprocal_rank", "mrr"]

    def __init__(self, threshold: float = 0.0, k: int = 10):
        """
        :class:`MeanReciprocalRankMetric`.

        :param threshold: The label threshold of relevance degree.
        """
        self._threshold = threshold
        self._k = k

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate reciprocal of the rank of the first relevant item.

        Example:
            >>> import numpy as np
            >>> y_pred = np.asarray([0.2, 0.3, 0.7, 1.0])
            >>> y_true = np.asarray([1, 0, 0, 0])
            >>> MeanReciprocalRank()(y_true, y_pred)
            0.25

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Mean reciprocal rank.
        """
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair[: self._k]):
            if label > self._threshold:
                return 1.0 / (idx + 1)
        return 0.0


class MeanReciprocalRankOptimal(Metric):
    """Mean reciprocal rank metric."""

    ALIAS = ["mean_reciprocal_rank_optimal", "mrr_opt"]

    def __init__(self, threshold: float = 0.0, k: int = 10):
        """
        :class:`MeanReciprocalRankMetric`.

        :param threshold: The label threshold of relevance degree.
        """
        self._threshold = threshold
        self._k = k

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate reciprocal of the rank of the first relevant item.

        Example:
            >>> import numpy as np
            >>> y_pred = np.asarray([0.2, 0.3, 0.7, 1.0])
            >>> y_true = np.asarray([1, 0, 0, 0])
            >>> MeanReciprocalRankOptimal()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Mean reciprocal rank.
        """
        coupled_pair = sort_and_couple(y_true, y_pred)
        if np.any(coupled_pair[: self._k, 0] > self._threshold):
            return 1.0
        return 0.0


class PairAccuracy(Metric):
    """Pair accuracy metric."""

    ALIAS = ["pair_accuracy", "pa"]

    def __init__(self) -> None:
        """
        :class:`PairAccuracy`.

        :param threshold: The label threshold of relevance degree.
        """

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate pairwise rank accuracy.

        Example:
            >>> import numpy as np
            >>> y_pred = np.asarray([0.3, 0.2, 0.7, 1.0])
            >>> y_true = np.asarray([1, 0, 0, 0])
            >>> PairAccuracy()(y_true, y_pred)
            0.3333333333333333

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Pair accuracy.
        """
        pair_mask = (y_true.reshape(-1, 1) - y_true.reshape(1, -1)) > 0
        if np.all(~pair_mask):
            return 0.0
        pred = y_pred.reshape(-1, 1) - y_pred.reshape(1, -1)
        pred = pred[pair_mask]
        acc: float = (pred > 0).sum() / len(pred)
        return acc


class TrecEval:
    """trec_eval wrapper."""

    def __init__(self, metrics: List[str], qrels: str, options: str = "") -> None:
        self._metrics = metrics
        self._qrels = qrels
        self._options = options

    def __call__(self, run: str) -> Dict[str, float]:
        metrics = " ".join([f"-m {x}" for x in self._metrics])
        args = f"trec_eval -q {self._options} {metrics} {self._qrels} {run}"

        proc = subprocess.run(args, shell=True, stdout=subprocess.PIPE, text=True)

        if run.endswith(".run"):
            eval_path = re.sub(r"\.run$", ".eval", run)
        else:
            eval_path = run + ".eval"
        with open(eval_path, "w") as f:
            f.write(proc.stdout)

        results = {}
        for line in proc.stdout.splitlines():
            splits = line.split()
            if splits[1] != "all":
                continue
            results[splits[0]] = float(splits[2])
        return results

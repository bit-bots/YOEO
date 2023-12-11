from __future__ import annotations

import numpy as np


class Metric:
    """
    Metric object providing usefule metrics based on a confusion matrix
    """

    def __init__(self, n_classes):
        self._n_classes = n_classes
        self._conf_matrix = np.zeros(shape=(n_classes, n_classes))

    def __add__(self, other: Metric):
        assert type(other) == Metric, "cannot add other than Metric"
        assert other._n_classes == self._n_classes, "Dimensions mismatch"

        m = Metric(self._n_classes)
        m._conf_matrix = self._conf_matrix + other._conf_matrix

        return m

    def _tp(self, class_id: int) -> int:
        return self._conf_matrix[class_id, class_id]

    def _fp(self, class_id: int) -> int:
        return np.sum(self._conf_matrix[class_id, :]) - self._conf_matrix[class_id, class_id]

    def _fn(self, class_id: int) -> int:
        return np.sum(self._conf_matrix[:, class_id]) - self._conf_matrix[class_id, class_id]

    def _tn(self, class_id: int) -> int:
        return np.sum(self._conf_matrix) - self._tp(class_id) - self._fp(class_id) - self._fn(class_id)


    def update(self, pred: int, target: int) -> None:
        self._conf_matrix[pred, target] += 1

    def merge(self, metric: Metric) -> None:
        self._conf_matrix += metric._conf_matrix

    def reset(self) -> None:
        self._conf_matrix = np.zeros(shape=(self._n_classes, self._n_classes))

    def get_conf_matrix(self) -> np.ndarray:
        return self._conf_matrix


    def ACC(self, class_id: int) -> float:
        denom = np.sum(self._conf_matrix)
        return (self._tp(class_id) + self._tn(class_id)) / denom if denom != 0 else float("nan")

    def mACC(self) -> float:
        return self._mean(self.ACC)

    def bACC(self, class_id: int) -> float:
        return (self.TPR(class_id) + self.TNR(class_id)) / 2

    def mbACC(self) -> float:
        return self._mean(self.bACC)

    def _mean(self, fun) -> float:
        return np.mean([fun(i) for i in range(self._n_classes)])

    def PREC(self, class_id: int) -> float:
        denom = (self._tp(class_id) + self._fp(class_id))
        return self._tp(class_id) / denom if denom != 0 else float("nan")

    def REC(self, class_id: int) -> float:
        return self.TPR(class_id)

    def F1(self, class_id: int) -> float:
        denom = (2 * self._tp(class_id) + self._fp(class_id) + self._fn(class_id))
        return 2 * self._tp(class_id) / denom if denom != 0 else float("nan")

    def TNR(self, class_id: int) -> float:
        denom = self._fp(class_id) + self._tn(class_id)
        return self._tn(class_id) / denom if denom != 0 else float("nan")

    def TPR(self, class_id: int) -> float:
        denom = (self._tp(class_id) + self._fn(class_id))
        return self._tp(class_id) / denom if denom != 0 else float("nan")
    
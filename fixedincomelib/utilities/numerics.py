import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6

    Node i "owns" the interval (axis1[i-1], axis1[i]]. With FLAT
    extrapolation, node 0 also owns everything to the left of axis1[0],
    and the last node owns everything to the right of axis1[-1].
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def _index(self, x: float) -> int:
        """Index of the node whose value applies at x (left-continuous)."""
        # side='left' returns the first i with axis1[i] >= x, so a point
        # exactly on a node uses that node's value (left-continuity).
        i = int(np.searchsorted(self.axis1_, x, side='left'))
        return min(i, self.length_ - 1)  # flat extrapolation on the right

    def interpolate(self, x: float) -> float:
        return float(self.values_[self._index(x)])

    def integrate(self, start_x: float, end_x: float) -> float:
        # The integral is linear in the node values: sum_i values[i] * w_i,
        # where w_i is the length of [start_x, end_x] inside node i's interval.
        weights = self.gradient_of_integrated_value_wrt_ordinate(start_x, end_x)
        return float(np.dot(self.values_, weights))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        # f(x) equals exactly one node value, so the gradient is one-hot.
        grad = np.zeros(self.length_)
        grad[self._index(x)] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        # Handle reversed limits: integral from b to a = -(integral from a to b).
        sign = 1.0
        a, b = start_x, end_x
        if b < a:
            a, b = b, a
            sign = -1.0

        weights = np.zeros(self.length_)
        for i in range(self.length_):
            lower = -np.inf if i == 0 else self.axis1_[i - 1]
            upper = np.inf if i == self.length_ - 1 else self.axis1_[i]
            overlap = min(b, upper) - max(a, lower)
            if overlap > 0:
                weights[i] = overlap
        return sign * weights



class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
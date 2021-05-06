from typing import Any, Callable

from pyrado.algorithms.stopping_criteria.stopping_criterion import StoppingCriterion


class AlwaysStopStoppingCriterion(StoppingCriterion):
    def _validate(self) -> bool:
        return True


class NeverStopStoppingCriterion(StoppingCriterion):
    def _validate(self) -> bool:
        return False


class CustomStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion_fn: Callable[[], bool]):
        super().__init__()
        self._criterion_fn = criterion_fn

    def _validate(self) -> bool:
        return self._criterion_fn()

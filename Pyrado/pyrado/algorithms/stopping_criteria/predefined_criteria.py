from typing import Any, Callable, Optional

from pyrado.algorithms.stopping_criteria.stopping_criterion import StoppingCriterion


class AlwaysStopStoppingCriterion(StoppingCriterion):
    def __repr__(self) -> str:
        return "AlwaysStopStoppingCriterion"

    def __str__(self) -> str:
        return "True"

    def _validate(self, algo) -> bool:
        return True


class NeverStopStoppingCriterion(StoppingCriterion):
    def __repr__(self) -> str:
        return "NeverStopStoppingCriterion"

    def __str__(self) -> str:
        return "False"

    def _validate(self, algo) -> bool:
        return False


class CustomStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion_fn: Callable[[Any], bool], name: Optional[str] = None):
        super().__init__()
        self._criterion_fn = criterion_fn
        self._name = name

    def __repr__(self) -> str:
        return f"CustomStoppingCriterion[_criterion_fn={repr(self._criterion_fn)}; name={self._name}]"

    def __str__(self) -> str:
        return "Custom" if self._name is None else self._name

    def _validate(self, algo) -> bool:
        return self._criterion_fn(algo)


class IterCountStoppingCriterion(StoppingCriterion):
    def __init__(self, max_iter: int):
        super().__init__()
        self._max_iter = max_iter

    def _validate(self, algo) -> bool:
        return algo.curr_iter >= self._max_iter


class SampleCountStoppingCriterion(StoppingCriterion):
    def __init__(self, max_iter: int):
        super().__init__()
        self._max_samples = max_iter

    def _validate(self, algo) -> bool:
        return algo.sample_count >= self._max_samples

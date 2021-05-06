from typing import Callable, Optional

from pyrado.algorithms.stopping_criteria.stopping_criterion import StoppingCriterion


class AlwaysStopStoppingCriterion(StoppingCriterion):
    def __repr__(self) -> str:
        return "AlwaysStopStoppingCriterion"

    def __str__(self) -> str:
        return "True"

    def _validate(self) -> bool:
        return True


class NeverStopStoppingCriterion(StoppingCriterion):
    def __repr__(self) -> str:
        return "NeverStopStoppingCriterion"

    def __str__(self) -> str:
        return "False"

    def _validate(self) -> bool:
        return False


class CustomStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion_fn: Callable[[], bool], name: Optional[str] = None):
        super().__init__()
        self._criterion_fn = criterion_fn
        self._name = name

    def __repr__(self) -> str:
        return f"CustomStoppingCriterion[_criterion_fn={repr(self._criterion_fn)}; name={self._name}]"

    def __str__(self) -> str:
        return "Custom" if self._name is None else self._name

    def _validate(self) -> bool:
        return self._criterion_fn()

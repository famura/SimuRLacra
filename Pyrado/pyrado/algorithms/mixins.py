from thirdParty.pytorch.torch.utils.data.sampler import Sampler
from typing import List

from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.step_sequence import StepSequence


class ExposedSampler:
    """A mixin class indicating that this algorithm exposes its sampler.

    Implementors: Save the used sampler in the `self._sampler` property.
    """

    @property
    def sampler(self) -> SamplerBase:
        return self._sampler

    @sampler.setter
    def sampler(self, value: SamplerBase) -> None:
        self._sampler = value

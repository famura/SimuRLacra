from typing import List

from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.step_sequence import StepSequence


class ExposedSampler:
    """A mixin class indicating that this algorithm exposes its sampler.

    Implementors: Save the used sampler in the `self.sampler` property.
    """

    def sample(self, *args, **kwargs) -> List[StepSequence]:
        """Calls the sample method of the algorithm's sampler.

        :param *args: Arguments to be forwarded to the sample method
        :param **kwargs: Keyword-Arguments to be forwarded to the sample method
        :return: A list of `StepSequence`s, which are generated according to the algorithms parameters (e.g. number of workers, rollout length, ...)
        :rtype: List[StepSequence]
        """
        return self.sampler.sample(*args, **kwargs)

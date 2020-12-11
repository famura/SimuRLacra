# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from init_args_serializer import Serializable
from typing import Mapping, Optional

import pyrado
from pyrado.environment_wrappers.base import EnvWrapperObs
from pyrado.environments.base import Env
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_processing import RunningNormalizer


class ObsNormWrapper(EnvWrapperObs, Serializable):
    """
    Environment wrapper which normalizes the observation space using the bounds from the environment or
    hard-coded bounds, such that all values are in range [-1, 1]
    """

    def __init__(
        self, wrapped_env: Env, explicit_lb: Mapping[str, float] = None, explicit_ub: Mapping[str, float] = None
    ):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param explicit_lb: dict to override the environment's lower bound; by default (`None`) this is ignored;
                            the keys are space labels, the values the new bound for that labeled entry
        :param explicit_ub: dict to override the environment's upper bound; by default (`None`) this is ignored;
                            the keys are space labels, the values the new bound for that labeled entry
        """
        Serializable._init(self, locals())
        super().__init__(wrapped_env)

        # Explicitly override the bounds if desired
        self.explicit_lb = explicit_lb
        self.explicit_ub = explicit_ub

        # Get the bounds of the inner observation space
        wos = self.wrapped_env.obs_space
        lb, ub = wos.bounds

        # Override the bounds if desired and store the result for usage in _process_obs
        self.ov_lb = ObsNormWrapper.override_bounds(lb, self.explicit_lb, wos.labels)
        self.ov_ub = ObsNormWrapper.override_bounds(ub, self.explicit_ub, wos.labels)

        # Check if the new bounds are valid
        if any(self.ov_lb == -pyrado.inf):
            raise pyrado.ValueErr(
                msg=f"At least one element of the lower bounds is (negative) infinite:\n"
                f"(overwritten) bound: {self.ov_lb}\nnames: {wos.labels}"
            )
        if any(self.ov_ub == pyrado.inf):
            raise pyrado.ValueErr(
                msg=f"At least one element of the upper bound is (positive) infinite:\n"
                f"(overwritten) bound: {self.ov_ub}\nnames: {wos.labels}"
            )

    @staticmethod
    def override_bounds(bounds: np.ndarray, override: Optional[Mapping[str, float]], names: np.ndarray) -> np.ndarray:
        """
        Override a given bound. This function is useful if some entries of the observation space have an infinite bound
        and/or you want to specify a certain bound

        :param bounds: bound to override
        :param override: value to override with
        :param names: label of the bound to override
        :return: new bound created from a copy of the old bound
        """
        if not override:
            return bounds
        # Override in copy of bounds
        bc = bounds.copy()
        for idx, name in np.ndenumerate(names):
            ov = override.get(name)
            if ov is not None:
                # Apply override
                bc[idx] = ov
            elif np.isinf(bc[idx]):
                # Report unbounded entry
                raise pyrado.ValueErr(
                    msg=f"The entry {name} of a bound is infinite and not overwritten." f"Cannot apply normalization!"
                )
            else:
                # Do nothing if ov is None
                pass
        return bc

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Normalize observation
        obs_norm = (obs - self.ov_lb) / (self.ov_ub - self.ov_lb) * 2 - 1
        return obs_norm

    def _process_obs_space(self, space: BoxSpace) -> BoxSpace:
        if not isinstance(space, BoxSpace):
            raise NotImplementedError("Only implemented ObsNormWrapper._process_obs_space() for BoxSpace!")

        # Return space with same shape but bounds from -1 to 1
        return BoxSpace(-np.ones(space.shape), np.ones(space.shape), labels=space.labels)


class ObsRunningNormWrapper(EnvWrapperObs, Serializable):
    """
    Environment wrapper which normalizes the observation space using the bounds from the environment or
    hard-coded bounds, such that all values are in range [-1, 1]
    """

    def __init__(self, wrapped_env: Env):
        """
        Constructor

        :param wrapped_env: environment to wrap
        """
        Serializable._init(self, locals())
        super().__init__(wrapped_env)

        # Explicitly override the bounds if desired
        self.normalizer = RunningNormalizer()

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.normalizer(obs)

    def _process_obs_space(self, space: BoxSpace) -> BoxSpace:
        if not isinstance(space, BoxSpace):
            raise NotImplementedError("Only implemented ObsRunningNormWrapper._process_obs_space() for BoxSpace!")

        # Return space with same shape but bounds from -1 to 1
        return BoxSpace(-np.ones(space.shape), np.ones(space.shape), labels=space.labels)

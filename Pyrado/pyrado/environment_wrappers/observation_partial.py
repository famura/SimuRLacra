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

import pyrado
from pyrado.environments.base import Env
from pyrado.environment_wrappers.base import EnvWrapperObs
from pyrado.spaces.base import Space
from init_args_serializer.serializable import Serializable


class ObsPartialWrapper(EnvWrapperObs, Serializable):
    """ Environment wrapper which creates a partial observation by masking certain elements """

    def __init__(self, wrapped_env: Env, mask: list = None, idcs: list = None, keep_selected: bool = False):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param mask: mask out array, entries with 1 are dropped (behavior can be inverted by keep_selected=True)
        :param idcs: indices to drop, ignored if mask is specified. If the observation space is labeled,
                     the labels can be used as indices.
        :param keep_selected: set to true to keep the mask entries with 1/the specified indices and drop the others
        """
        Serializable._init(self, locals())

        super(ObsPartialWrapper, self).__init__(wrapped_env)

        # Parse selection
        if mask is not None:
            # Use explicit mask
            mask = np.array(mask, dtype=np.bool)
            if not mask.shape == wrapped_env.obs_space.shape:
                raise pyrado.ShapeErr(given=mask, expected_match=wrapped_env.obs_space)
        else:
            # Parse indices
            assert idcs is not None, "Either mask or indices must be specified"
            mask = wrapped_env.obs_space.create_mask(idcs)
        # Invert if needed
        if keep_selected:
            self.keep_mask = mask
        else:
            self.keep_mask = np.logical_not(mask)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.keep_mask]

    def _process_obs_space(self, space: Space) -> Space:
        return space.subspace(self.keep_mask)

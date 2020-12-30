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
from typing import Dict, Tuple, Optional, Union, Mapping

import pyrado
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environments.sim_base import SimEnv
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env, all_envs, remove_env
from pyrado.utils.input_output import print_cbt


class DomainRandWrapper(EnvWrapper, Serializable):
    """ Base class for environment wrappers which call a `DomainRandomizer` to randomize the domain parameters """

    def __init__(self, wrapped_env: Union[SimEnv, EnvWrapper], randomizer: Optional[DomainRandomizer]):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param randomizer: `DomainRandomizer` object holding the probability distribution of all randomizable
                            domain parameters, pass `None` if you want to subclass wrapping another `DomainRandWrapper`
                            and use its randomizer
        """
        if not isinstance(inner_env(wrapped_env), SimEnv):
            raise pyrado.TypeErr(given=wrapped_env, expected_type=SimEnv)
        if not isinstance(randomizer, DomainRandomizer) and randomizer is not None:
            raise pyrado.TypeErr(given=randomizer, expected_type=DomainRandomizer)

        Serializable._init(self, locals())

        # Invoke EnvWrapper's constructor
        super().__init__(wrapped_env)

        self._randomizer = randomizer

    @property
    def randomizer(self) -> DomainRandomizer:
        return self._randomizer

    @randomizer.setter
    def randomizer(self, randomizer: DomainRandomizer):
        if not isinstance(randomizer, DomainRandomizer):
            raise pyrado.TypeErr(given=randomizer, expected_type=DomainRandomizer)
        self._randomizer = randomizer


class MetaDomainRandWrapper(DomainRandWrapper, Serializable):
    """
    Domain randomization wrapper which wraps another `DomainRandWrapper` to adapt its parameters,
    called domain distribution parameters.
    """

    def __init__(self, wrapped_rand_env: DomainRandWrapper, mapping: Mapping[int, Tuple[str, str]]):
        """
        Constructor

        :param wrapped_rand_env: randomized environment to wrap
        :param mapping: mapping from index of the numpy array (coming from the algorithm) to domain parameter name
                        (e.g. mass, length) and the domain distribution parameter (e.g. mean, std)

        .. code-block:: python

            # For the mapping arg use the this dict constructor
            ```
            m = {0: ('name1', 'parameter_type1'), 1: ('name2', 'parameter_type2')}
            ```
        """
        if not isinstance(wrapped_rand_env, DomainRandWrapper):
            raise pyrado.TypeErr(given=wrapped_rand_env, expected_type=DomainRandWrapper)

        Serializable._init(self, locals())

        # Invoke the DomainRandWrapper's constructor
        super().__init__(wrapped_rand_env, None)

        self.mapping = mapping

    @property
    def randomizer(self) -> DomainRandomizer:
        # Forward to the wrapped DomainRandWrapper
        return self._wrapped_env.randomizer

    @randomizer.setter
    def randomizer(self, dr: DomainRandomizer):
        # Forward to the wrapped DomainRandWrapper
        self._wrapped_env.randomizer = dr

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Forward to the wrapped DomainRandWrapper
        return self._wrapped_env.reset(init_state, domain_param)

    def adapt_randomizer(self, domain_distr_param_values: np.ndarray):
        # Check the input dimension and reshape if necessary
        if domain_distr_param_values.ndim == 1:
            pass
        elif domain_distr_param_values.ndim == 2:
            domain_distr_param_values = domain_distr_param_values.ravel()
        else:
            raise pyrado.ShapeErr(given=domain_distr_param_values, expected_match=(1,))

        # Reconfigure the wrapped environment's DomainRandomizer
        for i, value in enumerate(domain_distr_param_values):
            dp_name, ddp_name = self.mapping.get(i)
            self._wrapped_env.randomizer.adapt_one_distr_param(dp_name, ddp_name, value)


class DomainRandWrapperLive(DomainRandWrapper, Serializable):
    """
    Domain randomization wrapper which randomized the wrapped env at every reset.
    Thus every rollout is done with different domain parameters.
    """

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        if domain_param is None:
            # No explicit specification of domain parameters, so randomizer is called to draw a parameter dict
            self._randomizer.randomize(num_samples=1)
            domain_param = self._randomizer.get_params(fmt="dict", dtype="numpy")

        # Set the new domain parameters (and the initial sate) by calling the reset method of the wrapped env
        return self._wrapped_env.reset(init_state, domain_param)


class DomainRandWrapperBuffer(DomainRandWrapper, Serializable):
    """
    Domain randomization wrapper which randomized the wrapped env using a buffer of domain parameter sets.
    At every call of the reset method this wrapper cycles through that buffer.
    """

    def __init__(self, wrapped_env, randomizer: DomainRandomizer):
        """
        Constructor

        :param wrapped_env: environment to wrap around
        :param randomizer: `DomainRandomizer` object that manages the randomization
        """
        # Invoke the DomainRandWrapper's constructor
        super().__init__(wrapped_env, randomizer)

        self._ring_idx = None
        self._buffer = None

    @property
    def ring_idx(self) -> int:
        """ Get the buffer's index. """
        return self._ring_idx

    @ring_idx.setter
    def ring_idx(self, idx: int):
        """ Set the buffer's index. """
        assert isinstance(idx, int) and idx >= 0
        self._ring_idx = idx

    def fill_buffer(self, num_domains: int):
        """
        Fill the internal buffer with domains.

        :param num_domains: number of randomized domain parameter sets to store in the buffer
        """
        assert isinstance(num_domains, int) and num_domains >= 0
        self._randomizer.randomize(num_domains)
        self._buffer = self._randomizer.get_params(-1, fmt="list", dtype="numpy")
        self._ring_idx = 0

    @property
    def buffer(self):
        """ Get the domain parameter buffer. """
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: list):
        """
        Set the domain parameter buffer.
        Depends on the way the buffer has been saved, see the DomainRandomizer.get_params() arguments.

        :param buffer: list of dicts, each describing a domain or just one dict for one domain
        """
        if not (isinstance(buffer, list) or isinstance(buffer, dict)):
            raise pyrado.TypeErr(given=buffer, expected_type=[list, dict])
        self._buffer = buffer

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        if domain_param is None:
            # No explicit specification of domain parameters, so randomizer is requested
            if isinstance(self._buffer, dict):
                # The buffer consists of one domain parameter set
                domain_param = self._buffer
            elif isinstance(self._buffer, list):
                # The buffer consists of a list of domain parameter sets
                domain_param = self._buffer[self._ring_idx]
                self._ring_idx = (self._ring_idx + 1) % len(self._buffer)  # idx cycles over buffer
            else:
                raise pyrado.TypeErr(given=self._buffer, given_name="self._buffer", expected_type=[dict, list])
        else:
            # Explicit specification of domain parameters
            self._load_domain_param(domain_param)

        # Set the (new) domain params in the the wrapped env
        self._wrapped_env.domain_param = domain_param

        # Forward the rest to the reset method of the wrapped env
        return self._wrapped_env.reset(init_state, domain_param=None)

    def _get_state(self, state_dict: dict):
        super()._get_state(state_dict)
        state_dict["buffer"] = self._buffer
        state_dict["ring_idx"] = self._ring_idx

    def _set_state(self, state_dict: dict, copying: bool = False):
        super()._set_state(state_dict, copying)
        self._buffer = state_dict["buffer"]
        self._ring_idx = state_dict["ring_idx"]


def remove_all_dr_wrappers(env: SimEnv, verbose: bool = False):
    """
    Go through the environment chain and remove all wrappers of type `DomainRandWrapper` (and subclasses).

    :param env: env chain with domain randomization wrappers
    :param verbose: choose if status messages should be printed
    :return: env chain without domain randomization wrappers
    """
    while any(isinstance(subenv, DomainRandWrapper) for subenv in all_envs(env)):
        if verbose:
            print_cbt("Found domain randomization wrapper, trying to remove it.", "y", bright=True)
        try:
            env = remove_env(env, DomainRandWrapper)
            if verbose:
                print_cbt("Removed a domain randomization wrapper.", "g", bright=True)
        except Exception:
            raise RuntimeError("Could not remove the domain randomization wrapper!")
    return env

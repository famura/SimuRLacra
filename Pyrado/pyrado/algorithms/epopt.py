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

import joblib
import os.path as osp

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.sampling.cvar_sampler import CVaRSampler


class EPOpt(Algorithm):
    """
    Ensemble Policy Optimization (EPOpt)

    This algorithm wraps another algorithm on a very shallow level. It replaces the subroutine's sampler with a
    CVaRSampler, but does not have its own logger.

    .. seealso::
        [1] A. Rajeswaran, S. Ghotra, B. Ravindran, S. Levine, "EPOpt: Learning Robust Neural Network Policies using
        Model Ensembles", ICLR, 2017
    """

    name: str = 'epopt'

    def __init__(self,
                 env: Env,
                 subrtn: Algorithm,
                 skip_iter: int,
                 epsilon: float,
                 gamma: float = 1.):
        """
        Constructor

        :param env: same environment as the subroutine runs in. Only used for checking and saving the randomizer.
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param skip_iter: number of iterations for which all rollouts will be used (see prefix full)
        :param epsilon: quantile of rollouts that will be kept
        :param gamma: discount factor to compute the discounted return, default is 1 (no discount)
        """
        if not isinstance(subrtn, Algorithm):
            raise pyrado.TypeErr(given=subrtn, expected_type=Algorithm)
        if not typed_env(env, DomainRandWrapperLive):  # there is a domain randomization wrapper
            raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapperLive)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subrtn.save_dir, subrtn.max_iter, subrtn.policy, subrtn.logger)

        # Store inputs
        self._subrtn = subrtn
        self.epsilon = epsilon
        self.gamma = gamma
        self.skip_iter = skip_iter

        # Override the subroutine's sampler
        self._subrtn.sampler = CVaRSampler(
            self._subrtn.sampler,
            epsilon=1.,  # keep all rollouts until curr_iter = skip_iter
            gamma=self.gamma,
            min_rollouts=self._subrtn.sampler.min_rollouts,
            min_steps=self._subrtn.sampler.min_steps,
        )

        # Save initial randomizer
        joblib.dump(env.randomizer, osp.join(self.save_dir, 'randomizer.pkl'))

    @property
    def subroutine(self) -> Algorithm:
        return self._subrtn

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Activate the CVaR mechanism after skip_iter iterations
        if self.curr_iter == self.skip_iter:
            self._subrtn.sampler.epsilon = self.epsilon

        # Call subroutine
        self._subrtn.step(snapshot_mode, meta_info)

    def save_snapshot(self, meta_info: dict = None):
        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            if self.curr_iter == self.skip_iter:
                # Save the last snapshot before applying the CVaR
                self._subrtn.save_snapshot(meta_info=dict(prefix=f'iter_{self.skip_iter}'))
            else:
                self._subrtn.save_snapshot(meta_info=None)
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._subrtn.load_snapshot(ld, meta_info)
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

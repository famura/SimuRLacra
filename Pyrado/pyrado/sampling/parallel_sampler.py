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

import pickle
import sys
from init_args_serializer import Serializable
from tqdm import tqdm
from typing import List
import torch.multiprocessing as mp

from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler import SamplerBase


def _ps_init(G, env, policy):
    """ Store pickled (and thus copied) environment as well as policy. """
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)


def _ps_update_policy(G, state):
    """ Update policy state_dict. """
    G.policy.load_state_dict(state)


def _ps_sample_one(G):
    """
    Sample one rollout and return step count if counting steps, rollout count (1) otherwise.
    This function is used when a minimum number of steps was given.
    """
    ro = rollout(G.env, G.policy)
    return ro, len(ro)


def _ps_run_one(G, num):
    """ Sample one rollout. This function is used when a minimum number of rollouts was given. """
    return rollout(G.env, G.policy)


class ParallelSampler(SamplerBase, Serializable):
    """ Class for sampling from multiple environments in parallel """

    def __init__(self,
                 env,
                 policy,
                 num_envs: int,
                 *,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 seed: int = None):
        """
        Constructor

        :param env: environment to sample from
        :param policy: policy to act in the environment (can also be an exploration strategy)
        :param num_envs: number of parallel samplers
        :param min_rollouts: minimum number of complete rollouts to sample.
        :param min_steps: minimum total number of steps to sample.
        :param seed: seed value for the random number generators, pass `None` for no seeding
        """
        Serializable._init(self, locals())
        super().__init__(min_rollouts=min_rollouts, min_steps=min_steps)

        self.env = env
        self.policy = policy

        # Set method to spawn if using cuda
        if self.policy.device == 'cuda':
            mp.set_start_method('spawn', force=True)

        # Create parallel pool. We use one thread per env because it's easier.
        self.pool = SamplerPool(num_envs)

        if seed is not None:
            self.pool.set_seed(seed)

        # Distribute environments. We use pickle to make sure a copy is created for n_envs=1
        self.pool.invoke_all(_ps_init, pickle.dumps(self.env), pickle.dumps(self.policy))

    def reinit(self, env=None, policy=None):
        """ Re-initialize the sampler. """
        # Update env and policy if passed
        if env is not None:
            self.env = env
        if policy is not None:
            self.policy = policy

        # Always broadcast to workers
        self.pool.invoke_all(_ps_init, pickle.dumps(self.env), pickle.dumps(self.policy))

    def sample(self) -> List[StepSequence]:
        """ Do the sampling according to the previously given environment, policy, and number of steps/rollouts. """
        # Update policy's state
        self.pool.invoke_all(_ps_update_policy, self.policy.state_dict())

        # Collect samples
        with tqdm(leave=False, file=sys.stdout, desc='Sampling',
                  unit='steps' if self.min_steps is not None else 'rollouts') as pb:

            if self.min_steps is None:
                # Only minimum number of rollouts given, thus use run_map
                return self.pool.run_map(_ps_run_one, range(self.min_rollouts), pb)
            else:
                # Minimum number of steps given, thus use run_collect (automatically handles min_runs=None)
                return self.pool.run_collect(
                    self.min_steps,
                    _ps_sample_one,
                    collect_progressbar=pb,
                    min_runs=self.min_rollouts
                )[0]

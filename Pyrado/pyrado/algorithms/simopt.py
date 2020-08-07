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
import numpy as np
import os
import os.path as osp
import torch as to
from shutil import copyfile
from tabulate import tabulate
from typing import Sequence

import pyrado
from pyrado.algorithms.actor_critic import ActorCritic
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.sysid_as_rl import SysIdByEpisodicRL
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environments.quanser.base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.input_output import print_cbt


class SimOpt(Algorithm):
    """
    Simulation Optimization (SimOpt)

    .. note::
        A candidate is a set of parameter values for the domain parameter distribution and its value is the
        discrepancy between the simulated and real observations (based on a weighted metric).

    .. seealso::
        [1] Y. SimOpt, A. Handa, V. Makoviychuk, M. Macklin, J. Issac, N.D. Ratliff, D. Fox, "Closing the Sim-to-Real
        Loop: Adapting Simulation Randomization with Real World Experience", ICRA, 2020
    """

    name: str = 'simopt'
    iteration_key: str = 'simopt_iteration'  # logger's iteration key

    def __init__(self,
                 save_dir: str,
                 env_sim: MetaDomainRandWrapper,
                 env_real: [RealEnv, EnvWrapper],
                 subrtn_policy: Algorithm,
                 subrtn_distr: SysIdByEpisodicRL,
                 max_iter: int,
                 num_eval_rollouts: int = 5,
                 thold_succ: float = pyrado.inf,
                 thold_succ_subrtn: float = -pyrado.inf,
                 warmstart: bool = True,
                 policy_param_init: to.Tensor = None,
                 valuefcn_param_init: to.Tensor = None,
                 subrtn_snapshot_mode: str = 'best'):
        """
        Constructor

        .. note::
            If you want to continue an experiment, use the `load_dir` argument for the `train` call. If you want to
            initialize every of the policies with a pre-trained policy parameters use `policy_param_init`.

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain
        :param subrtn_policy: algorithm which performs the policy / value-function optimization
        :param subrtn_distr: algorithm which TODO
        :param max_iter: maximum number of iterations
        :param num_eval_rollouts: number of rollouts in the target domain to estimate the return
        :param thold_succ: success threshold on the real system's return for BayRn, stop the algorithm if exceeded
        :param thold_succ_subrtn: success threshold on the simulated system's return for the subrtn, repeat the
                                      subrtn until the threshold is exceeded or the for a given number of iterations
        :param warmstart: initialize the policy parameters with the one of the previous iteration. This option has no
                          effect for initial policies and can be overruled by passing init policy params explicitly.
        :param policy_param_init: initial policy parameter values for the subrtn, set `None` to be random
        :param valuefcn_param_init: initial value function parameter values for the subrtn, set `None` to be random
        :param subrtn_snapshot_mode: snapshot mode for saving during training of the subrtn
        """
        if not isinstance(env_sim, MetaDomainRandWrapper):
            raise pyrado.TypeErr(given=env_sim, expected_type=MetaDomainRandWrapper)
        if not isinstance(subrtn_policy, Algorithm):
            raise pyrado.TypeErr(given=subrtn_policy, expected_type=Algorithm)
        if not isinstance(subrtn_distr, SysIdByEpisodicRL):
            raise pyrado.TypeErr(given=subrtn_distr, expected_type=SysIdByEpisodicRL)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, subrtn_policy.policy, logger=None)

        # Store the inputs and initialize
        self._env_sim = env_sim
        self._env_real = env_real
        self._subrtn_policy = subrtn_policy
        self._subrtn_distr = subrtn_distr
        self.cands = None  # history of domain distribution parameters, called phi in [1]
        self.cands_values = None  # history of domain distribution parameters' discrepancies, called D in [1, eq. (4)]
        self.policy_param_init = policy_param_init.detach() if policy_param_init is not None else None
        self.valuefcn_param_init = valuefcn_param_init.detach() if valuefcn_param_init is not None else None
        self.warmstart = warmstart
        self.num_eval_rollouts = num_eval_rollouts
        self.subrtn_snapshot_mode = subrtn_snapshot_mode
        self.thold_succ = to.tensor([thold_succ])
        self.thold_succ_subrtn = to.tensor([thold_succ_subrtn])
        self.max_subrtn_rep = 3  # number of tries to exceed thold_succ_subrtn during training in simulation

    def train_policy_sim(self, cand: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param cand: hyper-parameters for the domain parameter distribution coming from the domain randomizer
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: estimated return of the trained policy in the target domain
        """
        # Save the current candidate
        to.save(cand.view(-1), osp.join(self._save_dir, f'{prefix}_candidate.pt'))

        # Set the domain randomizer
        self._env_sim.adapt_randomizer(cand.numpy())

        # Reset the subroutine algorithm which includes resetting the exploration
        self._subrtn_policy.reset()

        if not self.warmstart or self._curr_iter == 0:
            # Reset the subrtn's policy (and value function)
            self._subrtn_policy.policy.init_param(self.policy_param_init)
            if isinstance(self._subrtn_policy, ActorCritic):
                self._subrtn_policy.critic.value_fcn.init_param(self.valuefcn_param_init)
            if self.policy_param_init is None:
                print_cbt('Learning the new solution from scratch', 'y')
            else:
                print_cbt('Learning the new solution given an initialization', 'y')

        elif self.warmstart and self._curr_iter > 0:
            # Continue from the previous policy (and value function)
            self._subrtn_policy.policy.load_state_dict(
                to.load(osp.join(self._save_dir, f'iter_{self._curr_iter - 1}_policy.pt')).state_dict()
            )
            if isinstance(self._subrtn_policy, ActorCritic):
                self._subrtn_policy.critic.value_fcn.load_state_dict(
                    to.load(osp.join(self._save_dir, f'iter_{self._curr_iter - 1}_valuefcn.pt')).state_dict()
                )
            print_cbt(f'Initialized the new solution with the results from iteration {self._curr_iter - 1}', 'y')

        # Train a policy in simulation using the subroutine
        self._subrtn_policy.train(snapshot_mode=self.subrtn_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        ros = self.eval_behav_policy(
            None, self._env_sim, self._subrtn_policy.policy, prefix, self.num_eval_rollouts
        )
        avg_ret_sim = to.mean(to.tensor([r.undiscounted_return() for r in ros]))
        return float(avg_ret_sim)

    def train_randomizer(self, rollouts_real: Sequence[StepSequence], prefix: str) -> float:
        """
        Train and evaluate the policy that parametrizes domain randomizer, such that the loss given by the instance of
        `SysIdByEpisodicRL` is minimized.

        :param rollouts_real: recorded real-world rollouts
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: average system identification loss
        """
        # Reset the subroutine algorithm which includes resetting the exploration
        self._subrtn_distr.reset()

        # Train the domain distribution fitter using the subroutine
        self._subrtn_distr.train(snapshot_mode=self.subrtn_snapshot_mode,
                                 meta_info=dict(rollouts_real=rollouts_real, prefix=prefix))

        return self.eval_ddp_policy(rollouts_real)

    def eval_ddp_policy(self, rollouts_real: Sequence[StepSequence]) -> float:
        """
        Evaluate the policy that fits the domain parameter distribution to the observed rollouts.

        :param rollouts_real: recorded real-world rollouts
        :return: average system identification loss
        """
        # Run rollouts with the best fitter domain parameter distribution
        assert self._env_sim.randomizer is self._subrtn_distr.subrtn.env.randomizer
        init_states_real = np.array([ro.rollout_info['init_state'] for ro in rollouts_real])
        rollouts_sim = self.eval_behav_policy(
            None, self._env_sim, self._subrtn_policy.policy, '', self.num_eval_rollouts, init_states_real
        )

        # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
        ros_real_tr, ros_sim_tr = self._subrtn_distr.truncate_rollouts(rollouts_real, rollouts_sim)
        assert len(ros_real_tr) == len(ros_sim_tr)
        assert all([np.allclose(r.rollout_info['init_state'], s.rollout_info['init_state'])
                    for r, s in zip(ros_real_tr, ros_sim_tr)])

        # Return the average the loss
        losses = [self._subrtn_distr.obs_dim_weight@self._subrtn_distr.loss_fcn(ro_r, ro_s)
                  for ro_r, ro_s in zip(ros_real_tr, ros_sim_tr)]
        return float(np.mean(losses))

    @staticmethod
    def eval_behav_policy(save_dir: [str, None],
                          env: [RealEnv, SimEnv, MetaDomainRandWrapper],
                          policy: Policy,
                          prefix: str,
                          num_rollouts: int,
                          init_states: [np.ndarray, None] = None,
                          seed: int = 1001) -> Sequence[StepSequence]:
        """
        Evaluate a policy on the target system (real-world platform).
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param init_states: pass the initial states of the real system to sync the simulation (mandatory in this case)
        :param seed: seed value for the random number generators, only used when evaluating in simulation
        :return: rollouts
        """
        if isinstance(env, RealEnv):
            input('Evaluating in the target domain. Hit any key to continue.')
        if save_dir is not None:
            print_cbt(f'Executing {prefix}_policy ...', 'c', bright=True)

        ros = []
        if isinstance(env, RealEnv):
            # Evaluate in the real world
            for i in range(num_rollouts):
                ros.append(rollout(env, policy, eval=True))

        elif isinstance(env, (SimEnv, MetaDomainRandWrapper)):
            if init_states is None:
                init_states = np.array([env.init_space.sample_uniform() for _ in range(num_rollouts)])
            if init_states.shape[0] != num_rollouts:
                raise pyrado.ValueErr(msg='Number of init states must match the number of rollouts!')

            # Evaluate in simulation
            for i in range(num_rollouts):
                # there can be other sources of randomness aside the domain parameters
                ros.append(rollout(env, policy, eval=True, seed=seed,
                                   reset_kwargs=dict(init_state=init_states[i, :])))

        else:
            raise pyrado.TypeErr(given=env, expected_type=[RealEnv, SimEnv, MetaDomainRandWrapper])

        if save_dir is not None:
            # Save the evaluation results
            rets_real = to.tensor([r.undiscounted_return() for r in ros])
            to.save(rets_real, osp.join(save_dir, f'{prefix}_returns_real.pt'))

            print_cbt('Target domain performance', bright=True)
            print(tabulate([['mean return', to.mean(rets_real).item()],
                            ['std return', to.std(rets_real)],
                            ['min return', to.min(rets_real)],
                            ['max return', to.max(rets_real)]]))

        return ros

    def step(self, snapshot_mode: str = 'latest', meta_info: dict = None):
        if self.cands is None:
            # First iteration, use the random policy parameters
            cand = self._subrtn_distr.policy.param_values.detach()
            print('cand ', cand)
        else:
            # Select the latest domain distribution parameter set
            cand = self.cands[-1, :]

        # Train and evaluate the behavioral policy, repeat if the resulting policy did not exceed the success threshold
        prefix = f'iter_{self._curr_iter}'
        wrapped_trn_fcn = until_thold_exceeded(
            self.thold_succ_subrtn.item(), self.max_subrtn_rep
        )(self.train_policy_sim)
        wrapped_trn_fcn(cand, prefix)

        # Evaluate the current policy in the target domain
        policy = to.load(osp.join(self._save_dir, f'{prefix}_policy.pt'))
        rollouts_real = self.eval_behav_policy(
            self._save_dir, self._env_real, policy, prefix, self.num_eval_rollouts, None)

        # Train and evaluate the policy which represents domain parameter distribution. Save the real-world rollouts.
        curr_cand_value = self.train_randomizer(rollouts_real, prefix)
        next_cand = self._subrtn_distr.best_policy_param.detach()

        # Logging
        self.cands = to.cat([self.cands, next_cand], dim=0)
        self.cands_values = to.cat([self.cands_values, to.tensor(curr_cand_value)], dim=0)
        self.make_snapshot(snapshot_mode='latest', meta_info=meta_info)  # only latest makes sense

    def save_snapshot(self, meta_info: dict = None):
        # The subroutines are saving their snapshots during their training
        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            joblib.dump(self._env_sim, osp.join(self._save_dir, 'env_sim.pkl'))
            joblib.dump(self._env_real, osp.join(self._save_dir, 'env_real.pkl'))
            to.save(self.cands, osp.join(self._save_dir, 'candidates.pt'))
            to.save(self.cands_values, osp.join(self._save_dir, 'candidates_values.pt'))
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subrtn!')

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir
        if not osp.isdir(ld):
            raise pyrado.ValueErr(msg='Given path is not a directory!')

        if meta_info is None:
            # Crawl through the given directory and check how many policies and candidates there are
            found_policies, found_cands = None, None
            for root, dirs, files in os.walk(ld):
                found_policies = [p for p in files if p.endswith('_policy.pt')]  # 'policy.pt' file should not be found
                found_cands = [c for c in files if c.endswith('_candidate.pt')]

            # Copy to the current experiment's directory. Not necessary if we are continuing in that directory.
            if ld != self._save_dir:
                for p in found_policies:
                    copyfile(osp.join(ld, p), osp.join(self._save_dir, p))
                for c in found_cands:
                    copyfile(osp.join(ld, c), osp.join(self._save_dir, c))

            if len(found_policies) > 0:
                # Load all found candidates to save them into a single tensor
                found_cands.sort()  # the order is important since it determines the rows of the tensor
                self.cands = to.stack([to.load(osp.join(ld, c)) for c in found_cands])
                to.save(self.cands, osp.join(self._save_dir, 'candidates.pt'))

                # Catch the case that the algorithm stopped before evaluating a sampled candidate
                if not len(found_policies) == len(found_cands):
                    print_cbt(f'Found {len(found_policies)} policies, but {len(found_cands)} candidates!', 'r')
                    n = len(found_cands) - len(found_policies)
                    delete = input('Delete the superfluous candidates? [y / any other]').lower() == 'y'
                    if n > 0 and delete:
                        # Delete the superfluous candidates
                        print_cbt(f'Candidates before:\n{self.cands.numpy()}', 'w')
                        self.cands = self.cands[:-n, :]
                        found_cands = found_cands[:-n]
                        to.save(self.cands, osp.join(self._save_dir, 'candidates.pt'))
                        print_cbt(f'Candidates after:\n{self.cands.numpy()}', 'c')
                    else:
                        raise pyrado.ShapeErr(msg=f'Found {len(found_policies)} policies,'
                                                  f'but {len(found_cands)} candidates!')

            else:
                # Redo it all
                print_cbt('No policies have been found. Basically starting from scratch.', 'y', bright=True)
                self.cands = None

            try:
                # Crawl through the load_dir and copy all previous evaluations.
                # Not necessary if we are continuing in that directory.
                if ld != self._save_dir:
                    for root, dirs, files in os.walk(load_dir):
                        [copyfile(osp.join(load_dir, c), osp.join(self._save_dir, c))
                         for c in files if c.endswith('_rollouts_real.pt')]

                # Get all previously done evaluations. If we don't find any, the exception is caught.
                found_evals = None
                for root, dirs, files in os.walk(ld):
                    found_evals = [v for v in files if v.endswith('_rollouts_real.pt')]
                found_evals.sort()  # the order is important since it determines the rows of the tensor

                # Reconstruct candidates_values.pt
                self.cands_values = to.empty(self.cands.shape[0])
                for i, fe in enumerate(found_evals):
                    # Get the return estimate from the raw evaluations as in eval_behav_policy()
                    if self.mc_estimator:
                        self.cands_values[i] = to.mean(to.load(osp.join(ld, fe)))
                    else:
                        self.cands_values[i] = to.from_numpy(bootstrap_ci(
                            to.load(osp.join(ld, fe)).numpy(), np.mean,
                            num_reps=1000, alpha=0.05, ci_sides=1, studentized=False)[1])

                if len(found_evals) < len(found_cands):
                    print_cbt(f'Found {len(found_evals)} real-world evaluation files but {len(found_cands)} candidates.'
                              f' Now evaluation the remaining ones.', 'c', bright=True)
                for i in range(len(found_cands) - len(found_evals)):
                    # Evaluate the current policy in the target domain
                    if len(found_evals) < self.num_init_cand:
                        prefix = f'init_{i + len(found_evals)}'
                    else:
                        prefix = f'iter_{i + len(found_evals) - self.num_init_cand}'
                    policy = to.load(osp.join(self._save_dir, f'{prefix}_policy.pt'))
                    self.cands_values[i + len(found_evals)] = self.eval_behav_policy(
                        self._save_dir, self._env_real, policy, self.mc_estimator, prefix, self.num_eval_rollouts
                    )
                to.save(self.cands_values, osp.join(self._save_dir, 'candidates_values.pt'))

                if len(found_cands) < self.num_init_cand:
                    print_cbt('Found less candidates than the number of initial candidates.', 'y')
                else:
                    self.initialized = True

            except (FileNotFoundError, RuntimeError):
                # If there are returns_real.pt files but len(found_policies) > 0 (was checked earlier),
                # then the initial policies have not been evaluated yet
                self.eval_init_policies()

            # Get current iteration count
            found_iter_policies = None
            for root, dirs, files in os.walk(ld):
                found_iter_policies = [p for p in files if p.endswith('_policy.pt')]

            self._curr_iter = len(found_iter_policies)  # continue with next

            # Initialize subroutines with previous iteration
            self._subrtn_policy.load_snapshot(ld, meta_info=dict(prefix=f'iter_{self._curr_iter - 1}'))
            self._subrtn_distr.load_snapshot(ld, meta_info=dict(prefix=f'iter_{self._curr_iter - 1}'))  # TODO

            # This is the case if we found iter_i_candidate.pt but not iter_i_rollouts_real.pt
            if self.cands.shape[0] == self.cands_values.shape[0] + 1:
                # Evaluate and save the latest candidate on the target system
                curr_cand_value = self.eval_behav_policy(self._save_dir, self._env_real, self._subrtn_policy.policy,
                                                         self.mc_estimator, prefix=f'iter_{self._curr_iter - 1}',
                                                         num_rollouts=self.num_eval_rollouts)
                self.cands_values = to.cat([self.cands_values, curr_cand_value.view(1)], dim=0)
                to.save(self.cands_values, osp.join(self._save_dir, 'candidates_values.pt'))

        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subrtn!')

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
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.algorithms.utils import until_thold_exceeded, save_prefix_suffix, load_prefix_suffix
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.quanser.base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
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
                 subrtn_distr: SysIdViaEpisodicRL,
                 max_iter: int,
                 num_eval_rollouts: int = 5,
                 thold_succ: float = pyrado.inf,
                 thold_succ_subrtn: float = -pyrado.inf,
                 warmstart: bool = True,
                 policy_param_init: to.Tensor = None,
                 valuefcn_param_init: to.Tensor = None,
                 subrtn_snapshot_mode: str = 'latest',
                 logger: StepLogger = None):
        """
        Constructor

        .. note::
            If you want to continue an experiment, use the `load_dir` argument for the `train` call. If you want to
            initialize every of the policies with a pre-trained policy parameters use `policy_param_init`.

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain
        :param subrtn_policy: algorithm which performs the optimization of the behavioral policy (and value-function)
        :param subrtn_distr: algorithm which performs the optimization of the domain parameter distribution policy
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
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env_sim, MetaDomainRandWrapper):
            raise pyrado.TypeErr(given=env_sim, expected_type=MetaDomainRandWrapper)
        if not isinstance(subrtn_policy, Algorithm):
            raise pyrado.TypeErr(given=subrtn_policy, expected_type=Algorithm)
        if not isinstance(subrtn_distr, SysIdViaEpisodicRL):
            raise pyrado.TypeErr(given=subrtn_distr, expected_type=SysIdViaEpisodicRL)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, subrtn_policy.policy, logger)

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

        # Save initial environments
        joblib.dump(self._env_sim, osp.join(self._save_dir, 'env_sim.pkl'))
        joblib.dump(self._env_real, osp.join(self._save_dir, 'env_real.pkl'))
        joblib.dump(self._subrtn_distr.policy.prior, osp.join(self._save_dir, 'prior.pkl'))

    def train_policy_sim(self, cand: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param cand: hyper-parameters for the domain parameter distribution (need be compatible with the randomizer)
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: estimated return of the trained policy in the target domain
        """
        # Save the current candidate
        to.save(cand.view(-1), osp.join(self._save_dir, f'{prefix}_candidate.pt'))

        # Set the domain randomizer
        self._env_sim.adapt_randomizer(cand.detach().cpu().numpy())

        # Reset the subroutine algorithm which includes resetting the exploration
        self._subrtn_policy.reset()

        # Do a warm start if desired
        self._subrtn_policy.init_modules(
            self.warmstart, policy_param_init=self.policy_param_init, valuefcn_param_init=self.valuefcn_param_init
        )

        # Train a policy in simulation using the subroutine
        self._subrtn_policy.train(snapshot_mode=self.subrtn_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        ros = self.eval_behav_policy(
            None, self._env_sim, self._subrtn_policy.policy, prefix, self.num_eval_rollouts
        )
        avg_ret_sim = to.mean(to.tensor([r.undiscounted_return() for r in ros]))
        return float(avg_ret_sim)

    def train_ddp_policy(self, rollouts_real: Sequence[StepSequence], prefix: str) -> float:
        """
        Train and evaluate the policy that parametrizes domain randomizer, such that the loss given by the instance of
        `SysIdViaEpisodicRL` is minimized.

        :param rollouts_real: recorded real-world rollouts
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: average system identification loss
        """
        # Reset the subroutine algorithm which includes resetting the exploration
        self._subrtn_distr.reset()

        # Train the domain distribution fitter using the subroutine
        self._subrtn_distr.train(snapshot_mode=self.subrtn_snapshot_mode,
                                 meta_info=dict(rollouts_real=rollouts_real, prefix=prefix))

        return SimOpt.eval_ddp_policy(
            rollouts_real, self._env_sim, self.num_eval_rollouts, self._subrtn_distr, self._subrtn_policy
        )

    @staticmethod
    def eval_ddp_policy(rollouts_real: Sequence[StepSequence],
                        env_sim: MetaDomainRandWrapper,
                        num_rollouts: int,
                        subrtn_distr: SysIdViaEpisodicRL,
                        subrtn_policy: Algorithm) -> float:
        """
        Evaluate the policy that fits the domain parameter distribution to the observed rollouts.

        :param rollouts_real: recorded real-world rollouts
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param num_rollouts: number of rollouts to collect on the target domain
        :param subrtn_distr: algorithm which performs the optimization of the domain parameter distribution policy
        :param subrtn_policy: algorithm which performs the optimization of the behavioral policy (and value-function)
        :return: average system identification loss
        """
        # Run rollouts in simulation with the same initial states as the real-world rollouts
        assert env_sim.randomizer is subrtn_distr.subrtn.env.randomizer
        init_states_real = np.array([ro.rollout_info['init_state'] for ro in rollouts_real])
        rollouts_sim = SimOpt.eval_behav_policy(
            None, env_sim, subrtn_policy.policy, '', num_rollouts, init_states_real
        )

        # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
        ros_real_tr, ros_sim_tr = SysIdViaEpisodicRL.truncate_rollouts(
            rollouts_real, rollouts_sim, replicate=False
        )
        assert len(ros_real_tr) == len(ros_sim_tr)
        assert all([np.allclose(r.rollout_info['init_state'], s.rollout_info['init_state'])
                    for r, s in zip(ros_real_tr, ros_sim_tr)])

        # Return the average the loss
        losses = [subrtn_distr.loss_fcn(ro_r, ro_s) for ro_r, ro_s in zip(ros_real_tr, ros_sim_tr)]
        return float(np.mean(np.asarray(losses)))

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
        :param num_rollouts: number of rollouts to collect on the target domain
        :param init_states: pass the initial states of the real system to sync the simulation (mandatory in this case)
        :param seed: seed value for the random number generators, only used when evaluating in simulation
        :return: rollouts
        """
        if save_dir is not None:
            print_cbt(f'Executing {prefix}_policy ...', 'c', bright=True)

        ros_real = []
        if isinstance(inner_env(env), RealEnv):
            # Evaluate in the real world
            for i in range(num_rollouts):
                ros_real.append(rollout(env, policy, eval=True))

        elif isinstance(inner_env(env), SimEnv):
            if init_states is None:
                init_states = np.array([env.init_space.sample_uniform() for _ in range(num_rollouts)])
            if init_states.shape[0] != num_rollouts:
                raise pyrado.ValueErr(msg='Number of init states must match the number of rollouts!')

            # Evaluate in simulation
            for i in range(num_rollouts):
                # there can be other sources of randomness aside the domain parameters
                ros_real.append(
                    rollout(env, policy, eval=True, seed=seed, reset_kwargs=dict(init_state=init_states[i, :]))
                )

        else:
            raise pyrado.TypeErr(given=env, expected_type=[RealEnv, SimEnv])

        if save_dir is not None:
            # Save the evaluation results
            save_prefix_suffix(ros_real, 'rollouts_real', 'pkl', save_dir, meta_info=dict(prefix=prefix))
            rets_real = to.tensor([r.undiscounted_return() for r in ros_real])
            save_prefix_suffix(rets_real, 'returns_real', 'pt', save_dir, meta_info=dict(prefix=prefix))

            print_cbt('Target domain performance', bright=True)
            print(tabulate([['mean return', to.mean(rets_real).item()],
                            ['std return', to.std(rets_real)],
                            ['min return', to.min(rets_real)],
                            ['max return', to.max(rets_real)]]))

        return ros_real

    def step(self, snapshot_mode: str = 'latest', meta_info: dict = None):
        if self._curr_iter == 0:
            # First iteration, use the policy parameters (initialized from a prior)
            cand = self._subrtn_distr.policy.transform_to_ddp_space(self._subrtn_distr.policy.param_values)  # clones
            self.cands = cand.unsqueeze(0)
        else:
            # Select the latest domain distribution parameter set
            assert isinstance(self.cands, to.Tensor)
            cand = self.cands[-1, :].clone()

        print_cbt(f'Current domain distribution parameters: {cand.detach().cpu().numpy()}', 'g')

        # Train and evaluate the behavioral policy, repeat if the resulting policy did not exceed the success threshold
        prefix = f'iter_{self._curr_iter}'
        wrapped_trn_fcn = until_thold_exceeded(
            self.thold_succ_subrtn.item(), self.max_subrtn_rep
        )(self.train_policy_sim)
        wrapped_trn_fcn(cand, prefix)

        # Save the latest behavioral policy explicitly
        self._subrtn_policy.save_snapshot()

        # Evaluate the current policy in the target domain
        policy = to.load(osp.join(self._save_dir, f'{prefix}_policy.pt'))
        rollouts_real = self.eval_behav_policy(
            self._save_dir, self._env_real, policy, prefix, self.num_eval_rollouts, None
        )

        if self._curr_iter == 0:
            # First iteration, also evaluate the random initialization
            self.cands_values = SimOpt.eval_ddp_policy(
                rollouts_real, self._env_sim, self.num_eval_rollouts, self._subrtn_distr, self._subrtn_policy
            )
            self.cands_values = to.tensor(self.cands_values).unsqueeze(0)

        # Train and evaluate the policy which represents domain parameter distribution. Save the real-world rollouts.
        curr_cand_value = self.train_ddp_policy(rollouts_real, prefix)

        # The next candidate is the current search distribution and not the best policy parameter set (which is saved)
        next_cand = self._subrtn_distr.policy.transform_to_ddp_space(self._subrtn_distr.policy.param_values)

        # Logging
        self.cands = to.cat([self.cands, next_cand.unsqueeze(0)], dim=0)
        self.cands_values = to.cat([self.cands_values, to.tensor(curr_cand_value).unsqueeze(0)], dim=0)
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
            raise pyrado.PathErr(given=ld)

        if meta_info is None:
            # Crawl through the given directory and check how many policies and candidates there are
            found_policies, found_ddp_policies, found_cands, found_evals = None, None, None, None
            for root, dirs, files in os.walk(ld):
                dirs.clear()  # prevents walk() from going into subdirectories
                found_policies = [p for p in files if
                                  p.endswith('_policy.pt') and not 'ddp' in p]  # 'policy.pt' file should not be found
                found_ddp_policies = [p for p in files if
                                      p.endswith('_ddp_policy.pt')]  # 'ddp_policy.pt' file should not be found
                found_cands = [c for c in files if c.endswith('_candidate.pt')]
                found_evals = [v for v in files if v.endswith('_rollouts_real.pkl')]

            # The order is important since it determines the rows of the tensor
            found_policies.sort()
            found_ddp_policies.sort()
            found_cands.sort()
            found_evals.sort()

            # Copy to the current experiment's directory. Not necessary if we are continuing in that directory.
            if ld != self._save_dir:
                for p in found_policies:
                    copyfile(osp.join(ld, p), osp.join(self._save_dir, p))
                for ddp in found_ddp_policies:
                    copyfile(osp.join(ld, ddp), osp.join(self._save_dir, ddp))
                for c in found_cands:
                    copyfile(osp.join(ld, c), osp.join(self._save_dir, c))
                for e in found_evals:
                    copyfile(osp.join(ld, e), osp.join(self._save_dir, e))
                try:
                    copyfile(osp.join(ld, 'candidates_values.pt'), osp.join(self._save_dir, 'candidates_values.pt'))
                except FileNotFoundError:
                    pass

            # Reconstruct candidates_values.pt
            try:
                self.cands_values = to.load(osp.join(ld, 'candidates_values.pt'))
            except FileNotFoundError:
                self.cands_values = to.tensor([])

            if len(found_policies) > 0:
                # Load all found candidates to save them into a single tensor
                self.cands = to.stack([to.load(osp.join(ld, c)) for c in found_cands])
                to.save(self.cands, osp.join(self._save_dir, 'candidates.pt'))

                # Catch the case that the algorithm stopped before evaluating a sampled candidate
                if not len(found_policies) == len(found_cands):
                    print_cbt(f'Found {len(found_policies)} policies, but {len(found_cands)} candidates!', 'r')
                    n = len(found_cands) - len(found_policies)
                    delete = input('Delete the superfluous candidates? [y / any other] ').lower() == 'y'
                    if n > 0 and delete:
                        # Delete the superfluous candidates
                        print_cbt(f'Candidates before:\n{self.cands.numpy()}', 'w')
                        self.cands = self.cands[:-n, :]
                        found_cands = found_cands[:-n]  # cut if used later
                        to.save(self.cands, osp.join(self._save_dir, 'candidates.pt'))
                        print_cbt(f'Candidates after:\n{self.cands.numpy()}', 'c')
                    else:
                        raise pyrado.ShapeErr(msg=f'Found {len(found_policies)} policies,'
                                                  f'but {len(found_cands)} candidates!')

                if len(found_evals) < self.cands.shape[0]:
                    print_cbt(f'Found {len(found_evals)} real-world evaluation files but {self.cands.shape[0]} '
                              f'candidates. Now evaluation the remaining ones.', 'c', bright=True)
                for i in range(self.cands.shape[0] - len(found_evals)):
                    # Evaluate the current policy in the target domain
                    prefix = f'iter_{i + len(found_evals)}'
                    policy = to.load(osp.join(self._save_dir, f'{prefix}_policy.pt'))
                    ros_real = self.eval_behav_policy(
                        self._save_dir, self._env_real, policy, prefix, self.num_eval_rollouts, None
                    )
                    save_prefix_suffix(ros_real, 'rollouts_real', 'pkl', self._save_dir, meta_info=dict(prefix=prefix))
                    rets_real = to.tensor([r.undiscounted_return() for r in ros_real])
                    save_prefix_suffix(rets_real, 'returns_real', 'pt', self._save_dir, meta_info=dict(prefix=prefix))

                if len(found_ddp_policies) != len(found_policies):
                    print_cbt(f'Found {len(found_ddp_policies)} ddp policies, but {len(found_policies)} policies. '
                              f'Now training the remaining one.', 'c', bright=True)
                    n = len(found_policies) - len(found_ddp_policies)
                    assert n == 1
                    prefix = f'iter_{len(found_ddp_policies)}'
                    ros_real = joblib.load(osp.join(self._save_dir, f'{prefix}_rollouts_real.pkl'))
                    curr_cand_value = self.train_ddp_policy(ros_real, prefix)
                    # self.cands_values = to.cat([self.cands_values, to.tensor(curr_cand_value).unsqueeze(0)], dim=0)

                if len(self.cands_values) < self.cands.shape[0]:
                    print_cbt(f'Found {len(self.cands_values)} candidates values but {self.cands.shape[0]} '
                              f'candidates. Now evaluation the remaining ones.', 'c', bright=True)
                    for i in range(len(self.cands_values) - self.cands.shape[0]):
                        prefix = f'iter_{i + len(found_evals)}'
                        rollouts_real = joblib.load(osp.join(ld, f'{prefix}_rollouts_real.pkl'))
                        self._env_sim.adapt_randomizer(self.cands[i, :].detach().cpu().numpy())
                        self._subrtn_policy._policy = to.load(osp.join(ld, f'iter_{i}_policy.pt'))
                        curr_cand_value = SimOpt.eval_ddp_policy(
                            rollouts_real, self._env_sim, self.num_eval_rollouts, self._subrtn_distr,
                            self._subrtn_policy
                        )
                        self.cands_values = to.cat([self.cands_values, to.tensor(curr_cand_value).unsqueeze(0)], dim=0)
                elif len(self.cands_values) > self.cands.shape[0]:
                    print_cbt(f'Found {len(self.cands_values)} candidates values but {self.cands.shape[0]} '
                              f'candidates!', 'r')
                    n = len(self.cands_values) - self.cands.shape[0]
                    delete = input('Delete the superfluous  values? [y / any other] ').lower() == 'y'
                    if n > 0 and delete:
                        # Delete the superfluous candidates
                        print_cbt(f'Candidates values before:\n{self.cands_values.numpy()}', 'w')
                        self.cands_values = self.cands_values[:-n]
                        to.save(self.cands_values, osp.join(self._save_dir, 'candidates_values.pt'))
                        print_cbt(f'Candidates after:\n{self.cands_values.numpy()}', 'c')

            else:
                # Redo it all
                print_cbt('No policies have been found. Basically starting from scratch.', 'y', bright=True)
                self.cands = None
                self.cands_values = None

            # Get current iteration count
            self._curr_iter = len(found_policies)  # continue with next

            # Initialize subroutines with previous iteration
            if self._curr_iter > 0:
                # self._subrtn_policy.load_snapshot(ld, meta_info=dict(prefix=f'iter_{self._curr_iter - 1}'))
                # self._subrtn_distr.load_snapshot(ld, meta_info=dict(prefix=f'iter_{self._curr_iter - 1}'))

                prefix = f'iter_{self._curr_iter - 1}'
                self._subrtn_policy._policy = load_prefix_suffix(
                    self._subrtn_policy._policy, 'policy', 'pt', self._save_dir, dict(prefix=prefix))
                self._subrtn_distr.subrtn._policy = load_prefix_suffix(
                    self._subrtn_distr.subrtn._policy, 'ddp_policy', 'pt', self._save_dir, dict(prefix=prefix))

        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subrtn!')

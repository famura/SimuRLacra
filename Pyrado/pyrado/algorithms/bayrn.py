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
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement, PosteriorMean
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from shutil import copyfile
from tabulate import tabulate

import pyrado
from pyrado.algorithms.actor_critic import ActorCritic
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environments.quanser.base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.sampling.rollout import rollout
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import UnitCubeProjector
from pyrado.utils.standardizing import standardize


class BayRn(Algorithm):
    """
    Bayesian Domain Randomization (BayRn)

    .. note::
        A candidate is a set of parameter values for the domain parameter distribution and its value is the
        (estimated) real-world return.

    .. seealso::
        F. Muratore, C. Eilers, M. Gienger, J. Peters, "Bayesian Domain Randomization for Sim-to-Real Transfer",
        arXiv, 2020
    """

    name: str = 'bayrn'
    iteration_key: str = 'bayrn_iteration'  # logger's iteration key

    def __init__(self,
                 save_dir: str,
                 env_sim: MetaDomainRandWrapper,
                 env_real: [RealEnv, EnvWrapper],
                 subrtn: Algorithm,
                 bounds: to.Tensor,
                 max_iter: int,
                 acq_fc: str,
                 acq_restarts: int,
                 acq_samples: int,
                 acq_param: dict = None,
                 mc_estimator: bool = True,
                 num_eval_rollouts_real: int = 5,
                 num_eval_rollouts_sim: int = 50,
                 num_init_cand: int = 5,
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
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param bounds: boundaries for inputs of randomization function, format: [lower, upper]
        :param max_iter: maximum number of iterations
        :param acq_fc: Acquisition Function
                       'UCB': Upper Confidence Bound (default $\beta = 0.1$)
                       'EI': Expected Improvement
                       'PI': Probability of Improvement
        :param acq_restarts: number of restarts for optimizing the acquisition function
        :param acq_samples: number of initial samples for optimizing the acquisition function
        :param acq_param: hyper-parameter for the acquisition function, e.g. $\beta$ for UCB
        :param mc_estimator: estimate the return with a sample average (`True`) or a lower confidence
                                     bound (`False`) obtained from bootstrapping
        :param num_eval_rollouts_real: number of rollouts in the target domain to estimate the return
        :param num_eval_rollouts_sim: number of rollouts in simulation to estimate the return after training
        :param num_init_cand: number of initial policies to train, ignored if `init_dir` is provided
        :param thold_succ: success threshold on the real system's return for BayRn, stop the algorithm if exceeded
        :param thold_succ_subrtn: success threshold on the simulated system's return for the subroutine, repeat the
                                      subroutine until the threshold is exceeded or the for a given number of iterations
        :param warmstart: initialize the policy parameters with the one of the previous iteration. This option has no
                          effect for initial policies and can be overruled by passing init policy params explicitly.
        :param policy_param_init: initial policy parameter values for the subroutine, set `None` to be random
        :param valuefcn_param_init: initial value function parameter values for the subroutine, set `None` to be random
        :param subrtn_snapshot_mode: snapshot mode for saving during training of the subroutine
        """
        assert isinstance(env_sim, MetaDomainRandWrapper)
        assert isinstance(subrtn, Algorithm)
        assert bounds.shape[0] == 2
        assert all(bounds[1] > bounds[0])

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, subrtn.policy, logger=None)

        # Store the inputs and initialize
        self._env_sim = env_sim
        self._env_real = env_real
        self._subrtn = subrtn
        self.bounds = bounds
        self.cand_dim = bounds.shape[1]
        self.cands = None  # called x in the context of GPs
        self.cands_values = None  # called y in the context of GPs
        self.argmax_cand = to.Tensor()
        self.mc_estimator = mc_estimator
        self.acq_fcn_type = acq_fc.upper()
        self.acq_restarts = acq_restarts
        self.acq_samples = acq_samples
        self.acq_param = acq_param
        self.policy_param_init = policy_param_init.detach() if policy_param_init is not None else None
        self.valuefcn_param_init = valuefcn_param_init.detach() if valuefcn_param_init is not None else None
        self.warmstart = warmstart
        self.num_eval_rollouts_real = num_eval_rollouts_real
        self.num_eval_rollouts_sim = num_eval_rollouts_sim
        self.subrtn_snapshot_mode = subrtn_snapshot_mode
        self.thold_succ = to.tensor([thold_succ])
        self.thold_succ_subrtn = to.tensor([thold_succ_subrtn])
        self.max_subrtn_rep = 3  # number of tries to exceed thold_succ_subrtn during training in simulation
        self.curr_cand_value = -pyrado.inf  # for the stopping criterion
        self.uc_normalizer = UnitCubeProjector(bounds[0, :], bounds[1, :])

        # Set the flag to run the initialization phase. This is overruled if load_snapshot is called.
        self.initialized = False
        if num_init_cand > 0:
            self.num_init_cand = num_init_cand
        else:
            raise pyrado.ValueErr(given=num_init_cand, g_constraint='0')

        # Save initial environments
        self.save_snapshot()

    def stopping_criterion_met(self) -> bool:
        return self.curr_cand_value > self.thold_succ

    def train_policy_sim(self, cand: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param cand: hyper-parameters for the domain parameter distribution coming from the domain randomizer
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: estimated return of the trained policy in the target domain
        """
        # Save the individual candidate
        to.save(cand.view(-1), osp.join(self._save_dir, f'{prefix}_candidate.pt'))

        # Set the domain randomizer given the hyper-parameters
        self._env_sim.adapt_randomizer(cand.numpy())

        # Reset the subroutine's algorithm which includes resetting the exploration
        self._subrtn.reset()

        if not self.warmstart or self._curr_iter == 0:
            # Reset the subroutine's policy (and value function)
            self._subrtn.policy.init_param(self.policy_param_init)
            if isinstance(self._subrtn, ActorCritic):
                self._subrtn.critic.value_fcn.init_param(self.valuefcn_param_init)
            if self.policy_param_init is None:
                print_cbt('Learning the new solution from scratch', 'y')
            else:
                print_cbt('Learning the new solution given an initialization', 'y')

        elif self.warmstart and self._curr_iter > 0:
            # Continue from the previous policy (and value function)
            self._subrtn.policy.load_state_dict(
                to.load(osp.join(self._save_dir, f'iter_{self._curr_iter - 1}_policy.pt')).state_dict()
            )
            if isinstance(self._subrtn, ActorCritic):
                self._subrtn.critic.value_fcn.load_state_dict(
                    to.load(osp.join(self._save_dir, f'iter_{self._curr_iter - 1}_valuefcn.pt')).state_dict()
                )
            print_cbt(f'Initialized the new solution with the results from iteration {self._curr_iter - 1}', 'y')

        # Train a policy in simulation using the subroutine
        self._subrtn.train(snapshot_mode=self.subrtn_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        avg_ret_sim = self.eval_policy(
            None, self._env_sim, self._subrtn.policy, self.mc_estimator, prefix, self.num_eval_rollouts_sim
        )
        return float(avg_ret_sim)

    def train_init_policies(self):
        """
        Initialize the algorithm with a number of random distribution parameter sets a.k.a. candidates specified by
        the user. Train a policy for every candidate. Finally, store the policies and candidates.
        """
        cands = to.empty(self.num_init_cand, self.cand_dim)
        for i in range(self.num_init_cand):
            print_cbt(f'Generating initial domain instance and policy {i + 1} of {self.num_init_cand} ...',
                      'g', bright=True)
            # Generate random samples within bounds
            cands[i, :] = (self.bounds[1, :] - self.bounds[0, :])*to.rand(self.bounds.shape[1]) + self.bounds[0, :]
            # Train a policy for each candidate, repeat if the resulting policy did not exceed the success threshold
            print_cbt(f'Randomly sampled the next candidate: {cands[i].numpy()}', 'g')
            wrapped_trn_fcn = until_thold_exceeded(
                self.thold_succ_subrtn.item(), self.max_subrtn_rep
            )(self.train_policy_sim)
            wrapped_trn_fcn(cands[i], prefix=f'init_{i}')

        # Save candidates into a single tensor (policy is saved during training or exists already)
        to.save(cands, osp.join(self._save_dir, 'candidates.pt'))
        self.cands = cands

    def eval_init_policies(self):
        """
        Execute the trained initial policies on the target device and store the estimated return per candidate.
        The number of initial policies to evaluate is the number of found policies.
        """
        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(self._save_dir):
            found_policies = [p for p in files if p.startswith('init_') and p.endswith('_policy.pt')]
            found_cands = [c for c in files if c.startswith('init_') and c.endswith('_candidate.pt')]
        if not len(found_policies) == len(found_cands):
            raise pyrado.ValueErr(msg='Found a different number of initial policies than candidates!')
        elif len(found_policies) == 0:
            raise pyrado.ValueErr(msg='No policies or candidates found!')

        num_init_cand = len(found_cands)
        cands_values = to.empty(num_init_cand)

        # Load all found candidates to save them into a single tensor
        found_cands.sort()  # the order is important since it determines the rows of the tensor
        cands = to.stack([to.load(osp.join(self._save_dir, c)) for c in found_cands])

        # Evaluate learned policies from random candidates on the target environment (real-world) system
        for i in range(num_init_cand):
            policy = to.load(osp.join(self._save_dir, f'init_{i}_policy.pt'))
            cands_values[i] = self.eval_policy(self._save_dir, self._env_real, policy, self.mc_estimator,
                                               prefix=f'init_{i}', num_rollouts=self.num_eval_rollouts_real)

        # Save candidates's and their returns into tensors (policy is saved during training or exists already)
        # to.save(cands, osp.join(self._save_dir, 'candidates.pt'))
        to.save(cands_values, osp.join(self._save_dir, 'candidates_values.pt'))
        self.cands, self.cands_values = cands, cands_values

        if isinstance(self._env_real, RealEnv):
            input('Evaluated in the target domain. Hit any key to continue.')

    @staticmethod
    def eval_policy(save_dir: [str, None],
                    env: [RealEnv, SimEnv, MetaDomainRandWrapper],
                    policy: Policy,
                    mc_estimator: bool,
                    prefix: str,
                    num_rollouts: int) -> to.Tensor:
        """
        Evaluate a policy on the target system (real-world platform).
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param mc_estimator: estimate the return with a sample average (`True`) or a lower confidence
                                     bound (`False`) obtained from bootrapping
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :return: estimated return in the target domain
        """
        if isinstance(env, RealEnv):
            input('Evaluating in the target domain. Hit any key to continue.')
        if save_dir is not None:
            print_cbt(f'Executing {prefix}_policy ...', 'c', bright=True)

        rets_real = to.zeros(num_rollouts)
        if isinstance(env, RealEnv):
            # Evaluate sequentially when conducting a sim-to-real experiment
            for i in range(num_rollouts):
                rets_real[i] = rollout(env, policy, eval=True, no_close=False).undiscounted_return()
        elif isinstance(env, (SimEnv, MetaDomainRandWrapper)):
            # Create a parallel sampler when conducting a sim-to-sim experiment
            sampler = ParallelSampler(env, policy, num_workers=1, min_rollouts=num_rollouts)
            ros = sampler.sample()
            for i in range(num_rollouts):
                rets_real[i] = ros[i].undiscounted_return()
        else:
            raise pyrado.TypeErr(given=env, expected_type=[RealEnv, SimEnv, MetaDomainRandWrapper])

        if save_dir is not None:
            # Save the evaluation results
            to.save(rets_real, osp.join(save_dir, f'{prefix}_returns_real.pt'))

            print_cbt('target domain performance', bright=True)
            print(tabulate([['mean return', to.mean(rets_real).item()],
                            ['std return', to.std(rets_real)],
                            ['min return', to.min(rets_real)],
                            ['max return', to.max(rets_real)]]))

        if mc_estimator:
            return to.mean(rets_real)
        else:
            return to.from_numpy(bootstrap_ci(rets_real.numpy(), np.mean,
                                              num_reps=1000, alpha=0.05, ci_sides=1, studentized=False)[1])

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if not self.initialized:
            # Start initialization phase
            self.train_init_policies()
            self.eval_init_policies()
            self.initialized = True

        # Normalize the input data and standardize the output data
        cands_norm = self.uc_normalizer.project_to(self.cands)
        cands_values_stdized = standardize(self.cands_values).unsqueeze(1)

        # Create and fit the GP model
        gp = SingleTaskGP(cands_norm, cands_values_stdized)
        gp.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        print_cbt('Fitted the GP.', 'g')

        # Acquisition functions
        if self.acq_fcn_type == 'UCB':
            acq_fcn = UpperConfidenceBound(gp, beta=self.acq_param.get('beta', 0.1), maximize=True)
        elif self.acq_fcn_type == 'EI':
            acq_fcn = ExpectedImprovement(gp, best_f=cands_values_stdized.max().item(), maximize=True)
        elif self.acq_fcn_type == 'PI':
            acq_fcn = ProbabilityOfImprovement(gp, best_f=cands_values_stdized.max().item(), maximize=True)
        else:
            raise pyrado.ValueErr(given=self.acq_fcn_type, eq_constraint="'UCB', 'EI', 'PI'")

        # Optimize acquisition function and get new candidate point
        cand, acq_value = optimize_acqf(
            acq_function=acq_fcn,
            bounds=to.stack([to.zeros(self.cand_dim), to.ones(self.cand_dim)]),
            q=1,
            num_restarts=self.acq_restarts,
            raw_samples=self.acq_samples
        )
        next_cand = self.uc_normalizer.project_back(cand)
        print_cbt(f'Found the next candidate: {next_cand.numpy()}', 'g')
        self.cands = to.cat([self.cands, next_cand], dim=0)
        to.save(self.cands, osp.join(self._save_dir, 'candidates.pt'))

        # Train and evaluate a new policy, repeat if the resulting policy did not exceed the success threshold
        prefix = f'iter_{self._curr_iter}'
        wrapped_trn_fcn = until_thold_exceeded(
            self.thold_succ_subrtn.item(), self.max_subrtn_rep
        )(self.train_policy_sim)
        wrapped_trn_fcn(cand, prefix)

        # Evaluate the current policy in the target domain
        policy = to.load(osp.join(self._save_dir, f'{prefix}_policy.pt'))
        self.curr_cand_value = self.eval_policy(
            self._save_dir, self._env_real, policy, self.mc_estimator, prefix, self.num_eval_rollouts_real
        )

        self.cands_values = to.cat([self.cands_values, self.curr_cand_value.view(1)], dim=0)
        to.save(self.cands_values, osp.join(self._save_dir, 'candidates_values.pt'))

        # Store the argmax after training and evaluating
        curr_argmax_cand = BayRn.argmax_posterior_mean(self.cands, self.cands_values.unsqueeze(1),
                                                       self.uc_normalizer, self.acq_restarts, self.acq_samples)
        self.argmax_cand = to.cat([self.argmax_cand, curr_argmax_cand], dim=0)
        to.save(self.argmax_cand, osp.join(self._save_dir, 'candidates_argmax.pt'))

        self.make_snapshot(snapshot_mode, float(to.mean(self.cands_values)), meta_info)

    def save_snapshot(self, meta_info: dict = None):
        # Policies (and value functions) are saved by the subroutine in train_policy_sim()
        if meta_info is None:
            # This instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env_sim, osp.join(self._save_dir, 'env_sim.pkl'))
            joblib.dump(self._env_real, osp.join(self._save_dir, 'env_real.pkl'))
            to.save(self.bounds, osp.join(self._save_dir, 'bounds.pt'))
            to.save(self._subrtn.policy, osp.join(self._save_dir, 'policy.pt'))
            if isinstance(self._subrtn, ActorCritic):
                to.save(self._subrtn.critic.value_fcn, osp.join(self._save_dir, 'valuefcn.pt'))
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir
        if not osp.isdir(ld):
            raise pyrado.ValueErr(msg='Given path is not a directory!')

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._env_sim = joblib.load(osp.join(ld, 'env_sim.pkl'))
            self._env_real = joblib.load(osp.join(ld, 'env_real.pkl'))

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
                # Assuming not even the training of the initial policies has not been finished. Redo it all.
                print_cbt('No policies have been found. Basically starting from scratch.', 'c')
                self.train_init_policies()
                self.eval_init_policies()
                self.initialized = True

            try:
                # Crawl through the load_dir and copy all done evaluations.
                # Not necessary if we are continuing in that directory.
                if ld != self._save_dir:
                    for root, dirs, files in os.walk(load_dir):
                        [copyfile(osp.join(load_dir, c), osp.join(self._save_dir, c)) for c in files
                         if c.endswith('_returns_real.pt')]

                # Get all previously done evaluations. If we don't find any, the exception is caught.
                found_evals = None
                for root, dirs, files in os.walk(ld):
                    found_evals = [v for v in files if v.endswith('_returns_real.pt')]
                found_evals.sort()  # the order is important since it determines the rows of the tensor

                # Reconstruct candidates_values.pt
                self.cands_values = to.empty(self.cands.shape[0])
                for i, fe in enumerate(found_evals):
                    # Get the return estimate from the raw evaluations as in eval_policy()
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
                    self.cands_values[i + len(found_evals)] = self.eval_policy(
                        self._save_dir, self._env_real, policy, self.mc_estimator, prefix,
                        self.num_eval_rollouts_real
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
                self.initialized = True

            # Get current iteration count
            found_iter_policies = None
            for root, dirs, files in os.walk(ld):
                found_iter_policies = [p for p in files if p.startswith('iter_') and p.endswith('_policy.pt')]

            if not found_iter_policies:
                self._curr_iter = 0
                # We don't need to init the subroutine since it will be reset for iteration 0 anyway
            else:
                self._curr_iter = len(found_iter_policies)  # continue with next

                # Initialize subroutine with previous iteration
                self._subrtn.load_snapshot(ld, meta_info=dict(prefix=f'iter_{self._curr_iter - 1}'))

                # Evaluate and save the latest candidate on the target system.
                # This is the case if we found iter_i_candidate.pt but not iter_i_returns_real.pt
                if self.cands.shape[0] == self.cands_values.shape[0] + 1:
                    curr_cand_value = self.eval_policy(self._save_dir, self._env_real, self._subrtn.policy,
                                                       self.mc_estimator, prefix=f'iter_{self._curr_iter - 1}',
                                                       num_rollouts=self.num_eval_rollouts_real)
                    self.cands_values = to.cat([self.cands_values, curr_cand_value.view(1)], dim=0)
                    to.save(self.cands_values, osp.join(self._save_dir, 'candidates_values.pt'))

                    if isinstance(self._env_real, RealEnv):
                        input('Evaluated in the target domain. Hit any key to continue.')

        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

    @staticmethod
    def argmax_posterior_mean(cands: to.Tensor,
                              cands_values: to.Tensor,
                              uc_normalizer: UnitCubeProjector,
                              num_restarts: int,
                              num_samples: int) -> to.Tensor:
        """
        Compute the GP input with the maximal posterior mean.

        :param cands: candidates a.k.a. x
        :param cands_values: observed values a.k.a. y
        :param uc_normalizer: unit cube normalizer used during the experiments (can be recovered form the bounds)
        :param num_restarts: number of restarts for the optimization of the acquisition function
        :param num_samples: number of samples for the optimization of the acquisition function
        :return: un-normalized candidate with maximum posterior value a.k.a. x
        """
        # Normalize the input data and standardize the output data
        cands_norm = uc_normalizer.project_to(cands)
        cands_values_stdized = standardize(cands_values)

        # Create and fit the GP model
        gp = SingleTaskGP(cands_norm, cands_values_stdized)
        gp.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Find position with maximal posterior mean
        cand_norm, acq_value = optimize_acqf(
            acq_function=PosteriorMean(gp),
            bounds=to.stack([to.zeros_like(uc_normalizer.bound_lo), to.ones_like(uc_normalizer.bound_up)]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=num_samples
        )

        cand = uc_normalizer.project_back(cand_norm.detach())
        print_cbt(f'Converged to argmax of the posterior mean\n{cand.numpy()}', 'g', bright=True)
        return cand

    @staticmethod
    def train_argmax_policy(load_dir: str,
                            env_sim: MetaDomainRandWrapper,
                            subrtn: Algorithm,
                            num_restarts: int,
                            num_samples: int,
                            policy_param_init: to.Tensor = None,
                            valuefcn_param_init: to.Tensor = None,
                            subrtn_snapshot_mode: str = 'best') -> Policy:
        """
        Train a policy based on the maximizer of the posterior mean.

        :param load_dir: directory to load from
        :param env_sim: simulation environment
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param num_restarts: number of restarts for the optimization of the acquisition function
        :param num_samples: number of samples for the optimization of the acquisition function
        :param policy_param_init: initial policy parameter values for the subroutine, set `None` to be random
        :param valuefcn_param_init: initial value function parameter values for the subroutine, set `None` to be random
        :param subrtn_snapshot_mode: snapshot mode for saving during training of the subroutine
        :return: the final BayRn policy
        """
        # Load the required data
        cands = to.load(osp.join(load_dir, 'candidates.pt'))
        cands_values = to.load(osp.join(load_dir, 'candidates_values.pt')).unsqueeze(1)
        bounds = to.load(osp.join(load_dir, 'bounds.pt'))
        uc_normalizer = UnitCubeProjector(bounds[0, :], bounds[1, :])

        # Find the maximizer
        argmax_cand = BayRn.argmax_posterior_mean(cands, cands_values, uc_normalizer, num_restarts, num_samples)

        # Set the domain randomizer given the hyper-parameters
        env_sim.adapt_randomizer(argmax_cand.numpy())

        # Reset the subroutine's algorithm which includes resetting the exploration
        subrtn.reset()

        # Reset the subrtn's policy (and value function)
        subrtn.policy.init_param(policy_param_init)
        if isinstance(subrtn, ActorCritic):
            subrtn.critic.value_fcn.init_param(valuefcn_param_init)
        if policy_param_init is None:
            print_cbt('Learning the argmax solution from scratch', 'y')
        else:
            print_cbt('Learning the argmax solution given an initialization', 'y')

        subrtn.train(snapshot_mode=subrtn_snapshot_mode)
        return subrtn.policy

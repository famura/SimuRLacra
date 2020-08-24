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

import csv
import joblib
import numpy as np
import os.path as osp
import sys
import torch as to
from tqdm import tqdm
from warnings import warn

import pyrado
from pyrado.algorithms.actor_critic import ActorCritic
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.utils import typed_env
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sequences import *
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.utils.input_output import print_cbt


class SPOTA(Algorithm):
    r"""
    Simulation-based Policy Optimization with Probability Assessment (SPOTA)

    .. note::
        We use each domain parameter set $\xi_{j,nr}$ for $n_{\tau}$ rollouts.
        The candidate and the reference policies must have the same architecture!

    .. seealso::
        [1] F. Muratore, M. Gienger, J. Peters, "Assessing Transferability from Simulation to Reality for Reinforcement
        Learning", PAMI, 2019

        [2] W. Mak, D.P. Morton, and R.K. Wood, "Monte Carlo bounding techniques for determining solution quality in
        stochastic programs", Oper. Res. Lett., 1999
    """

    name: str = 'spota'
    iteration_key: str = 'spota_iteration'  # logger's iteration kay

    def __init__(self,
                 save_dir: str,
                 env: DomainRandWrapperBuffer,
                 subrtn_cand: Algorithm,
                 subrtn_refs: Algorithm,
                 max_iter: int,
                 alpha: float,
                 beta: float,
                 nG: int,
                 nJ: int,
                 ntau: int,
                 nc_init: int,
                 nr_init: int,
                 sequence_cand: callable,
                 sequence_refs: callable,
                 warmstart_cand: bool = False,
                 warmstart_refs: bool = True,
                 cand_policy_param_init: to.Tensor = None,
                 cand_critic_param_init: to.Tensor = None,
                 num_bs_reps: int = 1000,
                 studentized_ci: bool = False,
                 base_seed: int = None,
                 logger=None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param subrtn_cand: the algorithm that is called at every iteration of SPOTA to yield a candidate policy
        :param subrtn_refs: the algorithm that is called at every iteration of SPOTA to yield reference policies
        :param max_iter: maximum number of iterations that SPOTA algorithm runs.
                         Each of these iterations includes multiple iterations of the subroutine.
        :param alpha: confidence level for the upper confidence bound (UCBOG)
        :param beta: optimality gap threshold for training
        :param nG: number of reference solutions
        :param nJ: number of samples for Monte-Carlo approximation of the optimality gap
        :param ntau: number of rollouts per domain parameter set
        :param nc_init: initial number of domains for training the candidate solution
        :param nr_init: initial number of domains for training the reference solutions
        :param sequence_cand: mathematical sequence for the number of domains for training the candidate solution
        :param sequence_refs: mathematical sequence for the number of domains for training the reference solutions
        :param warmstart_cand: flag if the next candidate solution should be initialized with the previous one
        :param warmstart_refs: flag if the reference solutions should be initialized with the current candidate
        :param cand_policy_param_init: initial policy parameter values for the candidate, set None to be random
        :param cand_critic_param_init: initial critic parameter values for the candidate, set None to be random
        :param num_bs_reps: number of replications for the statistical bootstrap
        :param studentized_ci: flag if a student T distribution should be applied for the confidence interval
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        """
        if not typed_env(env, DomainRandWrapperBuffer):  # there is a domain randomization wrapper
            raise pyrado.TypeErr(msg='There must be a DomainRandWrapperBuffer in the environment chain.')
        if not isinstance(subrtn_cand, Algorithm):
            raise pyrado.TypeErr(given=subrtn_cand, expected_type=Algorithm)
        if not isinstance(subrtn_refs, Algorithm):
            raise pyrado.TypeErr(given=subrtn_refs, expected_type=Algorithm)

        # Call Algorithm's constructor without specifying the policy
        super().__init__(save_dir, max_iter, None, logger)

        # Get the randomized environment (recommended to make it the most outer one in the chain)
        self._env_dr = typed_env(env, DomainRandWrapperBuffer)

        # Candidate and reference solutions, and optimality gap
        self.Gn_diffs = None
        self.ucbog = pyrado.inf  # upper confidence bound on the optimality gap
        self._subrtn_cand = subrtn_cand
        self._subrtn_refs = subrtn_refs
        assert id(self._subrtn_cand) != id(self._subrtn_refs)
        assert id(self._subrtn_cand.policy) != id(self._subrtn_refs.policy)
        assert id(self._subrtn_cand.expl_strat) != id(self._subrtn_refs.expl_strat)

        # Store the hyper-parameters
        self.alpha = alpha
        self.beta = beta
        self.warmstart_cand = warmstart_cand
        self.warmstart_refs = warmstart_refs
        self.cand_policy_param_init = cand_policy_param_init.detach() if cand_policy_param_init is not None else None
        self.cand_critic_param_init = cand_critic_param_init.detach() if cand_critic_param_init is not None else None
        self.nG = nG
        self.nJ = nJ
        self.ntau = ntau
        self.nc_init = nc_init
        self.nr_init = nr_init
        self.seq_cand = sequence_cand
        self.seq_ref = sequence_refs
        self.num_bs_reps = num_bs_reps
        self.studentized_ci = studentized_ci
        self.base_seed = np.random.randint(low=10000) if base_seed is None else base_seed

        # Save initial environment and randomizer
        joblib.dump(env, osp.join(self.save_dir, 'init_env.pkl'))
        joblib.dump(env.randomizer, osp.join(self.save_dir, 'randomizer.pkl'))

    @property
    def subroutines(self) -> dict:
        return dict(cand=self._subrtn_cand, refs=self._subrtn_refs)

    def _adapt_batch_size(self, subroutine: Algorithm, n: int):
        """
        Adapt the number of dynamics transitions (steps or rollouts) of the subroutines according to the number of
        domains that is used in the current iteration of SPOTA.
        """
        if isinstance(subroutine, ParameterExploring):
            # Subclasses of ParameterExploring sample num_rollouts_per_param complete rollouts per iteration
            subroutine.sampler.num_rollouts_per_param = self.ntau*n

        elif isinstance(subroutine, ActorCritic):
            # The PPO sampler can either sample a minimum number of rollouts or steps
            if subroutine.sampler.min_steps is not None:
                subroutine.min_steps = self.ntau*n*self._env_dr.max_steps
                subroutine.sampler.set_min_count(min_steps=self.ntau*n*self._env_dr.max_steps)
            if subroutine.sampler.min_rollouts is not None:
                subroutine.min_rollouts = self.ntau*n
                subroutine.sampler.set_min_count(min_rollouts=self.ntau*n)
        else:
            raise NotImplementedError(f'No _adapt_batch_size method found for class {type(subroutine)}!')

    def stopping_criterion_met(self) -> bool:
        """
        Check if the upper confidence bound on the optimality gap is smaller than the specified threshold.
        .. note:: The UCBOG is equal to zero if all optimality gap samples are negative.
        """
        if self.ucbog != 0 and self.ucbog < self.beta:
            print_cbt(f'UCBOG is below specified threshold: {self.ucbog} < {self.beta}', 'g', bright=True)
            return True
        else:
            return False

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Candidate solution
        nc, _ = self.seq_cand(self.nc_init, self._curr_iter, dtype=int)
        self._adapt_batch_size(self._subrtn_cand, nc)
        self._compute_candidate(nc)

        # Reference solutions
        nr, _ = self.seq_ref(self.nr_init, self._curr_iter, dtype=int)
        self._adapt_batch_size(self._subrtn_refs, nr)
        self._compute_references(nr, self.nG)

        # Estimate the upper confidence bound on the optimality gap
        self._estimate_ucbog(nr)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, meta_info)

    def _compute_candidate(self, nc: int):
        """
        Train and save one candidate solution to a pt-file

        :param nc: number of domains used for training the candidate solution
        """
        # Do a warm start if desired
        self._subrtn_cand.init_modules(
            self.warmstart_cand, suffix='cand',
            policy_param_init=self.cand_policy_param_init,
            valuefcn_param_init=self.cand_critic_param_init
        )

        # Sample sets of physics params xi_{1}, ..., xi_{nc}
        self._env_dr.fill_buffer(nc)
        env_params_cand = self._env_dr.randomizer.get_params()
        joblib.dump(env_params_cand, osp.join(self._save_dir, f'iter_{self._curr_iter}_env_params_cand.pkl'))
        print('Randomized parameters of for the candidate solution:')
        print_domain_params(env_params_cand)

        # Reset the subroutine algorithm which includes resetting the exploration
        self._subrtn_cand.reset()
        print('Reset candidate exploration noise.')

        pol_param_before = self._subrtn_cand.policy.param_values.clone()
        if isinstance(self._subrtn_cand, ActorCritic):
            # Set dropout and batch normalization layers to training mode
            self._subrtn_cand.critic.value_fcn.train()
            critic_param_before = self._subrtn_cand.critic.value_fcn.param_values.clone()

        # Solve the (approx) stochastic program SP_nc for the sampled physics parameter sets
        print_cbt(f'\nIteration {self._curr_iter} | Candidate solution\n', 'c', bright=True)
        self._subrtn_cand.train(
            snapshot_mode='best', meta_info=dict(prefix=f'iter_{self._curr_iter}', suffix='cand')
        )

        if (self._subrtn_cand.policy.param_values == pol_param_before).all():
            warn("The candidate's policy parameters did not change during training!", UserWarning)
        if isinstance(self._subrtn_refs, ActorCritic):
            if (self._subrtn_cand.critic.value_fcn.param_values == critic_param_before).all():
                warn("The candidate's critic parameters did not change during training!", UserWarning)

        print_cbt('Learned an approx solution for SP_nc.\n', 'y')

    def _compute_references(self, nr: int, nG: int):
        """
        Train and save nG reference solutions to pt-files

        :param nr: number of domains used for training the reference solutions
        :param nG: number of reference solutions
        """
        # Loop to compute a distribution of optimality gaps via nG samples
        for k in range(nG):
            print_cbt(f'Iteration {self._curr_iter} | Reference solution {k + 1} of {nG}\n', 'c', bright=True)

            # Do a warm start if desired
            self._subrtn_refs.init_modules(
                self.warmstart_refs, suffix='cand',
                policy_param_init=self.cand_policy_param_init,
                valuefcn_param_init=self.cand_critic_param_init
            )

            # Sample new sets of physics params xi_{k,1}, ..., xi_{k,nr}
            self._env_dr.fill_buffer(nr)
            env_params_ref = self._env_dr.randomizer.get_params()
            joblib.dump(env_params_ref, osp.join(self._save_dir, f'iter_{self._curr_iter}_env_params_ref_{k}.pkl'))
            print('Randomized parameters of for the current reference solution:')
            print_domain_params(env_params_ref)

            # Reset the subroutine algorithm which includes resetting the exploration
            self._subrtn_refs.reset()
            print_cbt('Reset reference exploration noise.', 'y')

            pol_param_before = self._subrtn_refs.policy.param_values.clone()
            if isinstance(self._subrtn_refs, ActorCritic):
                # Set dropout and batch normalization layers to training mode
                self._subrtn_refs.critic.value_fcn.train()
                critic_param_before = self._subrtn_refs.critic.value_fcn.param_values.clone()

            # Solve the (approx) stochastic program SP_n for the samples physics parameter sets
            self._subrtn_refs.train(
                snapshot_mode='best', meta_info=dict(prefix=f'iter_{self._curr_iter}', suffix=f'ref_{k}')
            )

            if (self._subrtn_refs.policy.param_values == pol_param_before).all():
                warn("The reference's policy parameters did not change during training!", UserWarning)
            if isinstance(self._subrtn_refs, ActorCritic):
                if (self._subrtn_refs.critic.value_fcn.param_values == critic_param_before).all():
                    warn("The reference's critic parameters did not change during training!", UserWarning)

            print_cbt('Learned an approx solution for SP_n\n', 'y')

    def _eval_cand_and_ref_one_domain(self, i: int) -> tuple:
        """
        Evaluate the candidate and the k-th reference solution (see outer loop) in the i-th domain using nJ rollouts.

        :param i: index of the domain to evaluate in
        :return: average return values for the candidate and the k-th reference in the i-th domain
        """
        cand_ret_avg = 0.
        refs_ret_avg = 0.

        # Do nJ rollouts for each set of physics params
        for r in range(self.nJ):
            # Set the circular index for the particular realization
            self._env_dr.ring_idx = i
            # Do the rollout and collect the return
            ro_cand = rollout(self._env_dr, self._subrtn_cand.policy, eval=True, seed=self.base_seed + i*self.nJ + r)
            cand_ret_avg += ro_cand.undiscounted_return()

            # Set the circular index for the particular realization
            self._env_dr.ring_idx = i
            # Do the rollout and collect the return
            ro_ref = rollout(self._env_dr, self._subrtn_refs.policy, eval=True, seed=self.base_seed + i*self.nJ + r)
            refs_ret_avg += ro_ref.undiscounted_return()

        return cand_ret_avg/self.nJ, refs_ret_avg/self.nJ  # average over nJ seeds

    def _estimate_ucbog(self, nr: int):
        """
        Collect the returns with synchronized random seeds and estimate the pessimistic and optimistic bound.
        
        :param nr: number of domains used for training the reference solutions
        :return: upper confidence bound on the optimality gap (UCBOG)
        """
        # Init containers
        cand_rets = np.zeros((self.nG, nr))
        refs_rets = np.zeros((self.nG, nr))

        # Loop over all reference solutions
        for k in range(self.nG):
            print_cbt(f'Estimating the UCBOG | Reference {k + 1} of {self.nG} ...', 'c')
            # Load the domain parameters corresponding to the k-th reference solution
            env_params_ref = joblib.load(osp.join(self._save_dir, f'iter_{self._curr_iter}_env_params_ref_{k}.pkl'))
            self._env_dr.buffer = env_params_ref

            # Load the policies (makes a difference for snapshot_mode = best). They are set to eval mode by rollout()
            self._subrtn_cand.policy.load_state_dict(
                to.load(osp.join(self._save_dir, f'iter_{self._curr_iter}_policy_cand.pt')).state_dict()
            )
            self._subrtn_refs.policy.load_state_dict(
                to.load(osp.join(self._save_dir, f'iter_{self._curr_iter}_policy_ref_{k}.pt')).state_dict()
            )

            # Loop over all domain realizations of the reference solutions
            for i in tqdm(range(nr), total=nr, desc=f'Reference {k + 1}', unit='domains',
                          file=sys.stdout, leave=False):
                # Evaluate solutions
                cand_rets[k, i], refs_rets[k, i] = self._eval_cand_and_ref_one_domain(i)

                # Process negative optimality samples
                refs_rets = self._handle_neg_samples(cand_rets, refs_rets, k, i)

        # --------------
        # Optimality Gap
        # --------------

        # This is similar to the difference of the means that is used to calculate the optimality gap in eq. (9) in [2]
        self.Gn_diffs = np.subtract(refs_rets, cand_rets)  # optimistic bound - pessimistic bound; dim = nG x nr
        Gn_samples = np.mean(self.Gn_diffs, axis=1)  # dim = 1 x nr
        Gn_est = np.mean(Gn_samples)  # sample mean of the original (non-bootstrapped) samples

        ratio_neg_diffs = 1 - np.count_nonzero(self.Gn_diffs)/self.Gn_diffs.size  # assuming zero come from clipping

        print_cbt(f'diffs (optimistic - pessimistic bound):\n{self.Gn_diffs}', 'y')
        print_cbt(f'\n{100*ratio_neg_diffs}% of the diffs would have been negative and were set to 0\n',
                  'r', bright=True)

        if ratio_neg_diffs == 1:
            # All diffs are negative
            ci_bs = [0, float('inf')]  # such that the UCBOG comparison in stopping_criterion_met() does not break
            log_dict = {'Gn_est': np.NaN, 'UCBOG': np.NaN, 'ratio_neg_diffs': np.NaN}
        else:
            # Apply bootstrapping
            m_bs, ci_bs = bootstrap_ci(np.ravel(self.Gn_diffs), np.mean, self.num_bs_reps, self.alpha, 1,
                                       self.studentized_ci)
            print(f'm_bs: {m_bs}, ci_bs: {ci_bs}')
            print_cbt(f'\nOG (point estimate): {Gn_est} \nUCBOG: {ci_bs[1]}\n', 'y', bright=True)
            log_dict = {'Gn_est': Gn_est, 'UCBOG': ci_bs[1], 'ratio_neg_diffs': ratio_neg_diffs}

        # Log the optimality gap data
        mode = 'w' if self.curr_iter == 0 else 'a'
        with open(osp.join(self._save_dir, 'OG_log.csv'), mode, newline='') as csvfile:
            fieldnames = list(log_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if self.curr_iter == 0:
                writer.writeheader()
            writer.writerow(log_dict)

        # Store the current UCBOG estimated from all samples
        self.ucbog = ci_bs[1]

    def _handle_neg_samples(self, cand_rets: np.ndarray, refs_rets: np.ndarray, k: int, i: int) -> np.ndarray:
        """
        Process negative optimality gap samples by Looking at the other Reference Solutions

        :param cand_rets: array of the candidate's return values
        :param refs_rets: array of the references' return values
        :param k: index of the reference solution
        :param i: index of the domain
        :return refs_rets: if a better reference has been round the associated value will be overwritten
        """
        if refs_rets[k, i] < cand_rets[k, i]:
            print_cbt(
                f'\nReference {k + 1} is worse than the candidate on domain realization {i + 1}.\n'  # 1-based index
                'Trying to replace this reference at this realization with a different one', 'y')
            for other_k in range(self.nG):
                if other_k == k:
                    # Do nothing for the bad solution that brought us here
                    continue
                else:
                    # Load a reference solution different from the the k-th
                    other_ref = to.load(osp.join(self._save_dir, f'iter_{self._curr_iter}_policy_ref_{other_k}.pt'))

                    other_ref_ret = 0
                    for r in range(self.nJ):
                        # Set the same random seed
                        pyrado.set_seed(self.base_seed + i*self.nJ + r)
                        # Set the circular index for the particular realization
                        self._env_dr.ring_idx = i
                        # Do the rollout and collect the return
                        ro_other_ref = rollout(self._env_dr, other_ref)
                        other_ref_ret += ro_other_ref.undiscounted_return()/self.nJ  # average over nJ seeds
                    # Store the value if value is better
                    if other_ref_ret > refs_rets[k, i]:
                        refs_rets[k, i] = other_ref_ret
                        # If a better one was found, do not iterate over the remaining reference solutions
                        break

            if refs_rets[k, i] > cand_rets[k, i]:
                # Found a different reference that achieves a higher return that the candidate
                print_cbt('Successfully handled a negative OG sample', 'g')
            else:
                refs_rets[k, i] = cand_rets[k, i]  # forces optimality gap sample to be 0
                print_cbt('Unsuccessfully handled a negative OG sample: Set the value to 0', 'r')

        else:
            # Everything is as it should be
            pass

        return refs_rets

    def save_snapshot(self, meta_info: dict = None):
        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            np.save(osp.join(self._save_dir, f'iter_{self._curr_iter}_diffs.npy'), self.Gn_diffs)
            self._subrtn_cand.save_snapshot(meta_info=dict(suffix='cand'))
            self._subrtn_refs.save_snapshot(meta_info=dict(suffix='refs'))
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            self._subrtn_cand.load_snapshot(ld, meta_info=dict(suffix='cand'))
            self._subrtn_refs.load_snapshot(ld, meta_info=dict(suffix='refs'))
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

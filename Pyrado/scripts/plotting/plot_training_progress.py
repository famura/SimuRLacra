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

"""
Script to plot the training progress.
"""
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt
from pandas import DataFrame

from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.live_update import LiveFigureManager
from pyrado.plotting.curve import render_mean_std, render_lo_up_avg
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import read_csv_w_replace
from pyrado.utils.input_output import print_cbt


def _keys_in_columns(df: DataFrame, *cands, verbose: bool) -> bool:
    if any([c not in df.columns for c in cands]):
        if verbose:
            print_cbt(f'Did not find {list(cands)} in the data frame. Skipped the associated plot.')
        return False
    else:
        return True


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir
    file = osp.join(ex_dir, 'progress.csv')

    # Create plot manager that loads the progress data from the CSV into a Pandas data frame called df
    lfm = LiveFigureManager(file, read_csv_w_replace, args, update_interval=5)


    @lfm.figure('Number of Rollouts')
    def num_rollouts(fig, df, args):
        if not _keys_in_columns(df, 'num_rollouts', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.num_rollouts)), df.num_rollouts)
        plt.xlabel('iteration')
        plt.ylabel('number of rollouts')


    @lfm.figure('Average StepSequence Length')
    def avg_rollout_len(fig, df, args):
        if not _keys_in_columns(df, 'avg_rollout_len', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.avg_rollout_len)), df.avg_rollout_len)
        plt.xlabel('iteration')
        plt.ylabel('rollout length')


    @lfm.figure('Return -- Average & Minimum & Maximum')
    def return_min_max_avg(fig, df, args):
        if not _keys_in_columns(df, 'avg_return', 'min_return', 'max_return', verbose=args.verbose):
            return False
        render_lo_up_avg(fig.gca(), np.arange(len(df.min_return)), df.min_return, df.max_return, df.avg_return,
                         x_label='iteration', y_label='return', curve_label='average')


    @lfm.figure('Return -- Average & Standard Deviation & Median (& Current)')
    def return_avg_median(fig, df, args):
        if not _keys_in_columns(df, 'avg_return', 'std_return', 'median_return', verbose=args.verbose):
            return False
        render_mean_std(fig.gca(), np.arange(len(df.avg_return)), df.avg_return, df.std_return,
                        x_label='iteration', y_label='return', curve_label='average')
        plt.plot(np.arange(len(df.median_return)), df.median_return, label='median')
        plt.xlabel('iteration')
        plt.ylabel('return')
        if _keys_in_columns(df, 'curr_policy_return', verbose=args.verbose):
            # If the algorithm is a subclass of ParameterExploring
            plt.plot(np.arange(len(df.curr_policy_return)), df.curr_policy_return, label='current')
        plt.legend(loc='lower right')


    @lfm.figure('Explained Variance (R^2)')
    def explained_variance(fig, df, args):
        if not _keys_in_columns(df, 'explained_var', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.explained_var)), df.explained_var)
        plt.xlabel('iteration')
        plt.ylabel('explained variance')
        plt.ylim(-1.1, 1.1)


    @lfm.figure("Exploration Strategy's Standard Deviation")
    def explstrat_std(fig, df, args):
        if not _keys_in_columns(df, 'avg_expl_strat_std', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.avg_expl_strat_std)), df.avg_expl_strat_std, label='average')
        if 'min_expl_strat_std' in df.columns and 'max_expl_strat_std' in df.columns:
            plt.plot(np.arange(len(df.min_expl_strat_std)), df.min_expl_strat_std, label='smallest')
            plt.plot(np.arange(len(df.max_expl_strat_std)), df.max_expl_strat_std, label='largest')
        plt.xlabel('iteration')
        plt.ylabel('exploration std')
        plt.legend(loc='best')


    @lfm.figure("Exploration Strategy's Entropy")
    def explstrat_entropy(fig, df, args):
        if not _keys_in_columns(df, 'expl_strat_entropy', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.expl_strat_entropy)), df.expl_strat_entropy)
        plt.xlabel('iteration')
        plt.ylabel('exploration entropy')


    @lfm.figure("Average (inclusive) KL Divergence")
    def kl_divergence(fig, df, args):
        if not _keys_in_columns(df, 'avg_KL_old_new_', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.avg_KL_old_new_)), df.avg_KL_old_new_)
        plt.xlabel('iteration')
        plt.ylabel('KL divergence')


    @lfm.figure('Smallest and Largest Magnitude Policy Parameter')
    def extreme_policy_params(fig, df, args):
        if not _keys_in_columns(df, 'min_mag_policy_param', 'max_mag_policy_param', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.min_mag_policy_param)), df.min_mag_policy_param, label='smallest')
        plt.plot(np.arange(len(df.max_mag_policy_param)), df.max_mag_policy_param, label='largest')
        plt.xlabel('iteration')
        plt.ylabel('parameter value')
        plt.legend(loc='best')


    @lfm.figure('Loss Before and After Update Step')
    def loss_before_after(fig, df, args):
        if not _keys_in_columns(df, 'loss_before', 'loss_after', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.loss_before)), df.loss_before, label='before')
        plt.plot(np.arange(len(df.loss_after)), df.loss_after, label='after')
        plt.xlabel('iteration')
        plt.ylabel('loss value')
        plt.legend(loc='best')


    @lfm.figure('Policy and Value Function Gradient L-2 Norm')
    def avg_grad_norm(fig, df, args):
        if not _keys_in_columns(df, 'avg_policy_grad_norm', 'avg_vfcn_grad_norm', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.avg_policy_grad_norm)), df.avg_policy_grad_norm, label='policy')
        plt.plot(np.arange(len(df.avg_vfcn_grad_norm)), df.avg_vfcn_grad_norm, label='V-fcn')
        plt.xlabel('iteration')
        plt.ylabel('gradient norm')
        plt.legend(loc='best')


    """ CVaR sampler specific """


    @lfm.figure('Full Average StepSequence Length')
    def full_avg_rollout_len(fig, df, args):
        if not _keys_in_columns(df, 'full_avg_rollout_len', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.full_avg_rollout_len)), df.full_avg_rollout_len)
        plt.xlabel('iteration')
        plt.ylabel('rollout length')


    @lfm.figure('Full Return -- Average & Minimum & Maximum')
    def full_return_min_max_avg(fig, df, args):
        if not _keys_in_columns(df, 'full_avg_return', 'full_min_return', 'full_max_return', verbose=args.verbose):
            return False
        render_lo_up_avg(fig.gca(), np.arange(len(df.full_min_return)),
                         df.full_min_return, df.full_max_return, df.full_avg_return,
                         x_label='iteration', y_label='return', curve_label='average')
        plt.legend(loc='lower right')


    @lfm.figure('Full Return -- Average & Median & Standard Deviation')
    def return_avg_median_std(fig, df, args):
        if not _keys_in_columns(df, 'full_avg_return', 'full_median_return', 'full_std_return', verbose=args.verbose):
            return False
        render_mean_std(fig.gca(), np.arange(len(df.full_avg_return)), df.full_avg_return, df.full_std_return,
                        x_label='iteration', y_label='full return', curve_label='average')
        plt.plot(np.arange(len(df.full_median_return)), df.full_median_return, label='median')
        plt.legend(loc='lower right')


    """ REPS specific """


    @lfm.figure('REPS Dual Parameter')
    def eta(fig, df, args):
        if not _keys_in_columns(df, 'eta', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.eta)), df.eta)
        plt.xlabel('iteration')
        plt.ylabel(r'$\eta$')


    @lfm.figure('Dual Loss Before and After Update Step')
    def loss_before_after(fig, df, args):
        if not _keys_in_columns(df, 'dual_loss_before', 'dual_loss_after', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.dual_loss_before)), df.dual_loss_before, label='before')
        plt.plot(np.arange(len(df.dual_loss_after)), df.dual_loss_after, label='after')
        plt.xlabel('iteration')
        plt.ylabel('loss value')
        plt.legend(loc='best')


    """ SAC specific """


    @lfm.figure('SAC Temperature Parameter')
    def eta_coeff(fig, df, args):
        if not _keys_in_columns(df, 'ent_coeff', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.ent_coeff)), df.ent_coeff)
        plt.xlabel('iteration')
        plt.ylabel(r'entropy coefficient $\alpha$')


    @lfm.figure('Q-function Losses')
    def sac_q1_q2_losses(fig, df, args):
        if not _keys_in_columns(df, 'Q1_loss', 'Q2_loss', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.Q1_loss)), df.Q1_loss, label='$Q_1$')
        plt.plot(np.arange(len(df.Q2_loss)), df.Q2_loss, label='$Q_2$')
        plt.xlabel('iteration')
        plt.ylabel('loss value')
        plt.legend(loc='best')


    @lfm.figure('Policy Loss')
    def sac_policy_loss(fig, df, args):
        if not _keys_in_columns(df, 'policy_loss', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.policy_loss)), df.policy_loss)
        plt.xlabel('iteration')
        plt.ylabel('loss value')


    @lfm.figure('Policy Loss')
    def avg_memory_reward(fig, df, args):
        if not _keys_in_columns(df, 'avg_memory_reward', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.avg_memory_reward)), df.avg_memory_reward)
        plt.xlabel('iteration')
        plt.ylabel('average reward in replay buffer')


    """ CEM specific """


    @lfm.figure('Return -- Importance Samples')
    def importance_samples(fig, df, args):
        if not _keys_in_columns(df, 'min_imp_samp_return', 'median_imp_samp_return', 'max_return',
                                verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.min_imp_samp_return)), df.min_imp_samp_return, label='min')
        plt.plot(np.arange(len(df.median_imp_samp_return)), df.median_imp_samp_return, label='median')
        plt.plot(np.arange(len(df.max_return)), df.max_return, label='max')
        plt.xlabel('iteration')
        plt.ylabel('return')
        plt.legend(loc='best')


    """ TSPred specific """


    @lfm.figure('Training and Testing Loss')
    def importance_samples(fig, df, args):
        if not _keys_in_columns(df, 'trn_loss', 'tst_loss', verbose=args.verbose):
            return False
        plt.plot(np.arange(len(df.trn_loss)), df.trn_loss, label='train')
        plt.plot(np.arange(len(df.tst_loss)), df.tst_loss, label='test')
        plt.yscale('log')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(loc='best')


    # Start update loop
    lfm.spin()

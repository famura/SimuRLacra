import numpy as np

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats

from pyrado.algorithms.inference.sbi_rollout_sampler import RealRolloutSamplerForSBI


class DelfiSummaryStatistics(BaseSummaryStats):
    """Moment based SummaryStats class for the Hodgkin-Huxley model

    Calculates summary statistics
    """
    def __init__(self, seed=None):
        """See SummaryStats.py for docstring"""
        super(DelfiSummaryStatistics, self).__init__(seed=seed)

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of StepSequences

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []
        for ro in repetition_list:
            stats.append(np.array(RealRolloutSamplerForSBI.bayessim_statistic(ro)))

        return np.array(stats)
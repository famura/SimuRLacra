How to create an algorithm
--------------------------

This file provides a scheme of how to write an algorithm in Pyrado.
We differentiate between step-based and episodic algorithms.
In general, the first type inherits from `Algorithm` and randomizes the actions, whereas the second type randomizes the
policy parameters and inherits from `ParameterExploring` (which also inherits from `Algorithm`).

**Please see `algorithms/ppo.py` (i.e. also `algorithms/actor_critic.py`) or `algorithms/sac.py` examples if you want to implement a step-based RL algorithm, or `algorithms/pepg.py` and `algorithms/nes.py` if you want to implement an episodic RL algorithm.**

Start by creating a new class which inherits from `Algorithm` or `ParameterExploring`.

.. code-block:: python
    from typing import Sequence

    import pyrado
    from pyrado.algorithms.base import Algorithm
    from pyrado.environments.base import Env
    from pyrado.logger.step import StepLogger
    from pyrado.sampling.step_sequence import StepSequence


    class MFA(Algorithm):
        """
        My Fancy Algorithm (MFO)

        TODO a detailed description ...
        """

        name: str = 'mfo'  # TODO define an acronym that is used for saving and loading (lower case string recommended)

        def __init__(self,
                     # required args for every algorithm (workarounds are possible)
                     save_dir: str,
                     env: Env,
                     policy: Policy,
                     max_iter: int,
                     # TODO args specific to your algorithm ...
                     logger: StepLogger = None):

            # Call Algorithm's constructor. This instantiates the default step logger which works in most cases.
            # If you want a logger that only logs every N steps, check out `pyrado/algorithms/sac.py`
            super().__init__(save_dir, max_iter, policy, logger)

            # TODO store the inputs

            # TODO create an exploration strategy

            # TODO create a sampler

            # TODO set up an optimizer

Your algorithm must implement a `step()` function that performs a single iteration of the algorithm.
This includes collecting the data, updating the parameters, and adding the metrics of interest to the logger.
Does not update the `curr_iter` attribute since this is done in the `train()` method of the base function.
If this algorithm is run as a subroutine of a meta-algorithm, `meta_info` contains a dict of information about the
current iteration of the meta-algorithm, else leave it to `None`. For examples of meta-algorithms see
`algorithms/spota.py` or `algorithms/bayern.py`.

.. code-block:: python

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # TODO sample data (e.g. steps or rollouts) using the sampler created in `__init__()`

        # TODO compute metrics and add them to the logger

        # TODO pass the sampled data to the algorithm's `update()` method

        # TODO save snapshot data

Moreover, it is recommended to do the parameter update into a separate `update()` method.
Doing this is not strictly necessary and `update()` can have different inputs. Usually it is a batch of rollouts.

.. code-block:: python

    def update(self, rollouts: Sequence[StepSequence]):
        # TODO compute stuff from the agent's experience, i.e. the rollouts

        # TODO apply some nasty hacks to make the theory work (looking at you gradient clipping)

        # TODO update the parameters of the policy and optionally the exploration strategy

        # TODO optionally add some logging

In most cases you also want to override the `reset()` function. The base version resets the exploration strategy,
the iteration counter, and optionally sets a random seed, so be sure to call it. In most cases there are more things
to reset (e.g. the sampler).

.. code-block:: python

    def reset(self, seed: int = None):
        # Call the Algorithm's reset function
        super().reset(seed)

        # TODO Re-initialize sampler in case env or policy changed

        # TODO reset variables custom to your algorithm

You can override `stopping_criterion_met()` to specify additional stopping criteria.
Any subclass of `Algorithm` will always stop if the `curr_iter` counter is equal to `max_iter`.

.. code-block:: python

        def stopping_criterion_met(self) -> bool:
            return False  # TODO specify a stopping criterion for your algorithm

The following functions are called for saving (every step) and loading. The base class `Algorithm` saves the policy.

.. code-block:: python

    def save_snapshot(self, meta_info: dict = None):
        # Call Algorithm's save method
        super().save_snapshot(meta_info)

        # TODO save what needs to be saved (specific to your algorithm)

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # By default load from the directory where the algorithm saved to
        ld = load_dir if load_dir is not None else self._save_dir

        # TODO load what needs to be loaded (specific to your algorithm)

This tutorial is not meant to be exhaustive, but to give you an intuition what needs to be done.
I suggest to have a look at the existing algorithms and get some inspiration.
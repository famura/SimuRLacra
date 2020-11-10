Pyrado's Documentation
======================

Pyrado is a framework for reinforcement learning tailored domain randomization based on PyTorch and numpy.
The implementations are focus on modularity rather than on performance.

Installation
------------
Please see the root-level `readme file of SimuRLacra <https://github.com/famura/SimuRLacra/blob/master/README.md>`_.

Where to start?
---------------
Check out the provided examples or run some of the existing scripts in `Pyrado/scripts`.

.. toctree::
   :hidden:
   :caption: Examples:
   :maxdepth: 1
   :glob:

   examples/*


.. toctree::
   :caption: Algorithms:
   :maxdepth: 1

   algorithms
   algorithms.episodic
   algorithms.step_based
   algorithms.meta
   exploration
   sampling


.. toctree::
   :caption: Environments:
   :maxdepth: 1

   environments
   environments.pysim
   environments.mujoco
   environments.rcspysim
   environments.quanser
   environments.barrett_wam
   environment_wrappers
   domain_randomization
   spaces
   tasks


.. toctree::
   :caption: Policies:
   :maxdepth: 1

   policies
   policies.feed_forward
   policies.recurrent
   policies.special


.. toctree::
   :caption: Utilities:
   :maxdepth: 1

   logger
   plotting
   utils


   examples

   tests

* :ref:`modindex`

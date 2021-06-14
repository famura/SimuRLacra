Pyrado's Documentation
======================

Pyrado is a toolbox for reinforcement learning tailored domain randomization based on PyTorch and numpy.
It is designed to be used with RcsPySim in the `SimuRLacra <https://github.com/famura/SimuRLacra>`_ framework.
The implementations are focus on modularity rather than on performance.

Installation
------------
Please choose one of the options described in the root-level `readme file of SimuRLacra <https://github.com/famura/SimuRLacra/blob/master/README.md>`_.

Where to start?
---------------
Check out the provided examples or run some of the existing scripts in `Pyrado/scripts`.

Content
-------

.. toctree::
   :hidden:
   :caption: Examples:
   :maxdepth: 1
   :glob:

   examples/*

.. toctree::
   :caption: Environments:
   :maxdepth: 1

   environments
   environments.barrett_wam
   environments.mujoco
   environments.pysim
   environments.quanser
   environments.rcspysim

.. toctree::
   :caption: Environment Wrappers:
   :maxdepth: 1

   environment_wrappers

.. toctree::
   :caption: Domain Randomization:
   :maxdepth: 1

   domain_randomization

.. toctree::
   :caption: Algorithms:
   :maxdepth: 1

   algorithms
   algorithms.episodic
   algorithms.step_based
   algorithms.meta
   algorithms.inference
   algorithms.regression
   algorithms.utils

.. toctree::
   :caption: Exploration:
   :maxdepth: 1

   exploration

.. toctree::
   :caption: Policies:
   :maxdepth: 1

   policies
   policies.feed_back
   policies.feed_forward
   policies.recurrent
   policies.special

.. toctree::
   :caption: Spaces:
   :maxdepth: 1

   spaces

.. toctree::
   :caption: Tasks & Rewards:
   :maxdepth: 1

   tasks

.. toctree::
   :caption: Sampling:
   :maxdepth: 1

   sampling

.. toctree::
   :caption: Logging:
   :maxdepth: 1

    logger

.. toctree::
   :caption: Plotting:
   :maxdepth: 1

   plotting

.. toctree::
   :caption: Utilities:
   :maxdepth: 1

   utils

* :ref:`modindex`

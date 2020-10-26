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
This file is found by pytest and contains fixtures (i.e., common defaults) that can be used for all tests.
"""
import pytest
import multiprocessing as mp

from pyrado.domain_randomization.domain_parameter import UniformDomainParam, NormalDomainParam, \
    MultivariateNormalDomainParam, DomainParam
from pyrado.environments.one_step.catapult import CatapultSim
from pyrado.environments.one_step.rosenbrock import RosenSim
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim, QQubeStabSim
from pyrado.policies.adn import ADNPolicy, pd_cubic
from pyrado.policies.dummy import DummyPolicy, IdlePolicy
from pyrado.policies.features import *
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.linear import LinearPolicy
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.policies.neural_fields import NFPolicy
from pyrado.policies.rnn import RNNPolicy, GRUPolicy, LSTMPolicy
from pyrado.policies.time import TimePolicy, TraceableTimePolicy
from pyrado.policies.two_headed import TwoHeadedFNNPolicy, TwoHeadedGRUPolicy


# set spawn method to spawn for parallel test runs
mp.set_start_method('spawn', force=True)

# Set default torch dtype globally to avoid inconsistent errors depending on the test run order
to.set_default_dtype(to.double)

# Check if RcsPySim, Bullet, and Vortex are available
try:
    import rcsenv
    from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate2DSim, BallOnPlate5DSim
    from pyrado.environments.rcspysim.box_lifting import BoxLiftingPosDSSim
    from pyrado.environments.rcspysim.box_shelving import BoxShelvingVelDSSim, BoxShelvingPosDSSim
    from pyrado.environments.rcspysim.mp_blending import MPBlendingSim
    from pyrado.environments.rcspysim.planar_3_link import Planar3LinkIKActivationSim, Planar3LinkTASim
    from pyrado.environments.rcspysim.planar_insert import PlanarInsertTASim, PlanarInsertIKActivationSim
    from pyrado.environments.rcspysim.quanser_qube import QQubeRcsSim
    from pyrado.environments.rcspysim.target_tracking import TargetTrackingSim

    m_needs_vortex = pytest.mark.skipif(
        not rcsenv.supportsPhysicsEngine('Vortex'),
        reason='Vortex physics engine is not supported in this setup.'
    )
    m_needs_bullet = pytest.mark.skipif(
        not rcsenv.supportsPhysicsEngine('Bullet'),
        reason='Bullet physics engine is not supported in this setup.'
    )
    m_needs_rcs = pytest.mark.skipif(False, reason='rcsenv can be imported.')

    m_needs_libtorch = pytest.mark.skipif(
        'torch' not in rcsenv.ControlPolicy.types, reason='Requires RcsPySim compiled locally with libtorch!'
    )

except (ImportError, ModuleNotFoundError):
    m_needs_vortex = pytest.mark.skip
    m_needs_bullet = pytest.mark.skip
    m_needs_rcs = pytest.mark.skip
    m_needs_libtorch = pytest.mark.skip

# Check if MuJoCo i.e. mujoco-py is available
try:
    import mujoco_py
    from pyrado.environments.mujoco.openai_half_cheetah import HalfCheetahSim
    from pyrado.environments.mujoco.openai_hopper import HopperSim
    from pyrado.environments.mujoco.wam import WAMBallInCupSim

    m_needs_mujoco = pytest.mark.skipif(False, reason='mujoco-py can be imported.')

except (ImportError, Exception):
    m_needs_mujoco = pytest.mark.skip(reason='mujoco-py is not supported in this setup.')

# Check if CUDA support is available
m_needs_cuda = pytest.mark.skipif(not to.cuda.is_available(), reason='CUDA is not supported in this setup.')


# --------------------
# Environment fixtures
# --------------------

@pytest.fixture(scope='function')
def env(request):
    if hasattr(request, 'param'):
        marker = request.param
    else:
        marker = request.node.get_closest_marker("env")
        if marker is not None:
            marker = marker.args[0]
    if marker is None:
        raise ValueError("No envs specified")
    else:
        return getattr(DefaultEnvs, marker)()


class DefaultEnvs:
    @staticmethod
    def default_cata():
        return CatapultSim(max_steps=1, example_config=False)

    @staticmethod
    def default_rosen():
        return RosenSim()

    @staticmethod
    def default_bob():
        return BallOnBeamSim(dt=0.01, max_steps=500)

    @staticmethod
    def default_omo():
        return OneMassOscillatorSim(dt=0.02, max_steps=300)

    @staticmethod
    def default_pend():
        return PendulumSim(dt=0.02, max_steps=400)

    @staticmethod
    def default_qbb():
        return QBallBalancerSim(dt=0.01, max_steps=500)

    @staticmethod
    def default_qcpst():
        return QCartPoleStabSim(dt=0.01, max_steps=300)

    @staticmethod
    def default_qcpsu():
        return QCartPoleSwingUpSim(dt=0.002, max_steps=8000)

    @staticmethod
    def default_qqst():
        return QQubeStabSim(dt=0.01, max_steps=500)

    @staticmethod
    def default_qqsu():
        return QQubeSwingUpSim(dt=0.004, max_steps=4000)

    @staticmethod
    @m_needs_bullet
    def default_bop2d_bt():
        return BallOnPlate2DSim(physicsEngine='Bullet', dt=0.01, max_steps=3000, checkJointLimits=False)

    @staticmethod
    @m_needs_vortex
    def default_bop2d_vx():
        return BallOnPlate2DSim(physicsEngine='Vortex', dt=0.01, max_steps=3000, checkJointLimits=False)

    @staticmethod
    @m_needs_bullet
    def default_bop5d_bt():
        return BallOnPlate5DSim(physicsEngine='Bullet', dt=0.01, max_steps=3000, checkJointLimits=False)

    @staticmethod
    @m_needs_vortex
    def default_bop5d_vx():
        return BallOnPlate5DSim(physicsEngine='Vortex', dt=0.01, max_steps=3000, checkJointLimits=False)

    @staticmethod
    @m_needs_bullet
    def default_p3l_ik_bt():
        return Planar3LinkIKActivationSim(
            physicsEngine='Bullet',
            dt=1/50.,
            max_steps=1000,
            max_dist_force=None,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeForceTorque=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=False,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeTaskSpaceDiscrepancy=True,
        )

    @staticmethod
    @m_needs_vortex
    def default_p3l_ik_vx():
        return Planar3LinkIKActivationSim(
            physicsEngine='Vortex',
            dt=1/50.,
            max_steps=1000,
            max_dist_force=None,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeForceTorque=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=False,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeTaskSpaceDiscrepancy=True,
        )

    @staticmethod
    @m_needs_bullet
    def default_p3l_ta_bt():
        return Planar3LinkTASim(
            physicsEngine='Bullet',
            dt=1/50.,
            max_steps=1000,
            max_dist_force=None,
            position_mps=True,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeForceTorque=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeTaskSpaceDiscrepancy=True,
            observeDynamicalSystemGoalDistance=True,
            observeDynamicalSystemDiscrepancy=True,
        )

    @staticmethod
    @m_needs_vortex
    def default_p3l_ta_vx():
        return Planar3LinkTASim(
            physicsEngine='Vortex',
            dt=1/50.,
            max_steps=1000,
            max_dist_force=None,
            position_mps=True,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeForceTorque=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeTaskSpaceDiscrepancy=True,
            observeDynamicalSystemGoalDistance=True,
            observeDynamicalSystemDiscrepancy=True,
        )

    @staticmethod
    @m_needs_vortex
    def default_pi_ik_6l_vx():
        return PlanarInsertIKActivationSim(
            physicsEngine='Vortex',
            graphFileName='gPlanarInsert6Link.xml',
            dt=1/50.,
            max_steps=500,
            max_dist_force=None,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeForceTorque=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeDynamicalSystemGoalDistance=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True,
        )

    @staticmethod
    @m_needs_bullet
    def default_pi_ik_5l_bt():
        return PlanarInsertIKActivationSim(
            physicsEngine='Bullet',
            graphFileName='gPlanarInsert5Link.xml',
            dt=1/50.,
            max_steps=500,
            max_dist_force=None,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeForceTorque=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeDynamicalSystemGoalDistance=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True,
        )

    @staticmethod
    @m_needs_bullet
    def default_pi_ta_6l_bt():
        return PlanarInsertTASim(
            physicsEngine='Bullet',
            graphFileName='gPlanarInsert6Link.xml',
            dt=1/50.,
            max_steps=500,
            max_dist_force=None,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeForceTorque=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeDynamicalSystemGoalDistance=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True,
        )

    @staticmethod
    @m_needs_vortex
    def default_pi_ta_5l_vx():
        return PlanarInsertTASim(
            physicsEngine='Vortex',
            graphFileName='gPlanarInsert5Link.xml',
            dt=1/50.,
            max_steps=500,
            max_dist_force=None,
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeForceTorque=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeDynamicalSystemGoalDistance=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True,
        )

    @staticmethod
    @m_needs_bullet
    def default_blpos_bt():
        return BoxLiftingPosDSSim(
            physicsEngine='Bullet',
            graphFileName='gBoxLifting_posCtrl.xml',
            dt=0.01,
            max_steps=15000,
            mps_left=None,
            mps_right=None,
            ref_frame='basket',
            collisionConfig={'file': 'collisionModel.xml'},
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeCurrentManipulability=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True
        )

    @staticmethod
    @m_needs_bullet
    def default_bspos_bt():
        return BoxShelvingPosDSSim(
            physicsEngine='Bullet',
            graphFileName='gBoxShelving_posCtrl.xml',  # gBoxShelving_posCtrl.xml or gBoxShelving_trqCtrl.xml
            dt=1/100.,
            max_steps=2000,
            fix_init_state=True,
            ref_frame='upperGoal',
            mps_left=None,
            mps_right=None,
            collisionConfig={'file': 'collisionModel.xml'},
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeForceTorque=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True,
            observeDynamicalSystemGoalDistance=True,
        )

    @staticmethod
    @m_needs_vortex
    def default_bspos_vx():
        return BoxShelvingPosDSSim(
            physicsEngine='Vortex',
            graphFileName='gBoxShelving_posCtrl.xml',  # gBoxShelving_posCtrl.xml or gBoxShelving_trqCtrl.xml
            dt=1/100.,
            max_steps=2000,
            fix_init_state=True,
            ref_frame='upperGoal',
            mps_left=None,
            mps_right=None,
            collisionConfig={'file': 'collisionModel.xml'},
            taskCombinationMethod='sum',
            checkJointLimits=True,
            collisionAvoidanceIK=True,
            observeVelocities=True,
            observeForceTorque=True,
            observeCollisionCost=True,
            observePredictedCollisionCost=True,
            observeManipulabilityIndex=True,
            observeDynamicalSystemDiscrepancy=True,
            observeTaskSpaceDiscrepancy=True,
            observeDynamicalSystemGoalDistance=True,
        )

    @staticmethod
    @m_needs_bullet
    def default_qqsurcs_bt():
        return QQubeRcsSim(physicsEngine='Bullet', dt=1/250., max_steps=3000)

    @staticmethod
    @m_needs_mujoco
    def default_cth():
        return HalfCheetahSim()

    @staticmethod
    @m_needs_mujoco
    def default_hop():
        return HopperSim()

    @staticmethod
    @m_needs_mujoco
    def default_wambic():
        return WAMBallInCupSim(num_dof=7, max_steps=1750)


# ---------------
# Policy fixtures
# ---------------


@pytest.fixture(scope='function')
def policy(request, env):
    selected_policy = request.param
    if selected_policy is None:
        raise ValueError("No policy specified")
    else:
        return getattr(DefaultPolicies, selected_policy)(env)


class DefaultPolicies:
    @staticmethod
    def default_fs():
        return FeatureStack([const_feat, identity_feat, squared_feat])

    @staticmethod
    def idle_policy(env):
        return IdlePolicy(env.spec)

    @staticmethod
    def dummy_policy(env):
        return DummyPolicy(env.spec)

    @staticmethod
    def linear_policy(env):
        return LinearPolicy(env.spec, DefaultPolicies.default_fs())

    @staticmethod
    def time_policy(env):
        def timefcn(t: float):
            return list(np.random.rand(env.spec.act_space.flat_dim))

        return TimePolicy(env.spec, dt=env.dt, fcn_of_time=timefcn)

    @staticmethod
    def tracetime_policy(env):
        def timefcn(t: float):
            return list(np.random.rand(env.spec.act_space.flat_dim))

        return TraceableTimePolicy(env.spec, dt=env.dt, fcn_of_time=timefcn)

    @staticmethod
    def fnn_policy(env):
        return FNNPolicy(env.spec, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)

    @staticmethod
    def rnn_policy(env):
        return RNNPolicy(env.spec, hidden_size=8, num_recurrent_layers=2, hidden_nonlin='tanh')

    @staticmethod
    def lstm_policy(env):
        return LSTMPolicy(env.spec, hidden_size=8, num_recurrent_layers=2)

    @staticmethod
    def gru_policy(env):
        return GRUPolicy(env.spec, hidden_size=8, num_recurrent_layers=2)

    @staticmethod
    def adn_policy(env):
        return ADNPolicy(env.spec, dt=env.dt, activation_nonlin=to.sigmoid, potentials_dyn_fcn=pd_cubic)

    @staticmethod
    def nf_policy(env):
        return NFPolicy(env.spec, dt=env.dt, hidden_size=5, mirrored_conv_weights=True, tau_learnable=True,
                        init_param_kwargs=dict(bell=True), kappa_learnable=True, potential_init_learnable=True)

    @staticmethod
    def thfnn_policy(env):
        return TwoHeadedFNNPolicy(env.spec, shared_hidden_sizes=[16, 16], shared_hidden_nonlin=to.relu)

    @staticmethod
    def thgru_policy(env):
        return TwoHeadedGRUPolicy(env.spec, shared_hidden_size=8, shared_num_recurrent_layers=1)


@pytest.fixture(scope='function')
def default_pert():
    return DomainRandomizer(
        NormalDomainParam(name='mass', mean=1.2, std=0.1, clip_lo=10, clip_up=100),
        UniformDomainParam(name='special', mean=0, halfspan=42, clip_lo=-7.4, roundint=True),
        NormalDomainParam(name='length', mean=4, std=0.6, clip_up=50.1),
        UniformDomainParam(name='time_delay', mean=13, halfspan=6, clip_up=17, roundint=True),
        MultivariateNormalDomainParam(name='multidim', mean=10*to.ones((2,)), cov=2*to.eye(2), clip_up=11)
    )


@pytest.fixture(scope='function')
def default_dummy_randomizer():
    return DomainRandomizer(
        DomainParam(name='mass', mean=1.2),
        DomainParam(name='special', mean=0),
        DomainParam(name='length', mean=4),
        DomainParam(name='time_delay', mean=13)
    )


@pytest.fixture(scope='function')
def bob_pert():
    return DomainRandomizer(
        NormalDomainParam(name='g', mean=9.81, std=0.981, clip_lo=1e-3),
        NormalDomainParam(name='r_ball', mean=0.1, std=0.01, clip_lo=1e-3),
        NormalDomainParam(name='m_ball', mean=0.5, std=0.05, clip_lo=1e-3),
        NormalDomainParam(name='m_beam', mean=3.0, std=0.3, clip_lo=1e-3),
        NormalDomainParam(name='d_beam', mean=0.1, std=0.01, clip_lo=1e-3),
        NormalDomainParam(name='l_beam', mean=2.0, std=0.2, clip_lo=1e-3),
        UniformDomainParam(name='c_frict', mean=0.05, halfspan=0.05),
        UniformDomainParam(name='ang_offset', mean=0, halfspan=5.*np.pi/180)
    )

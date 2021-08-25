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
Storage for default a.k.a. nominal domain parameter values and default randomizers
"""
from typing import Dict, Tuple, Union

import numpy as np

import pyrado
from pyrado.domain_randomization.domain_parameter import BernoulliDomainParam, NormalDomainParam, UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.sim_base import SimEnv


DEFAULT_RANDOMIZER_REGISTRY = {}


def default_randomizer(env_module, env_class):
    """
    Register a default randomizer provider for a given environment type.
    The environment type is referenced by name to avoid eager loading of all environments when this module is used.

    :usage:
    .. code-block:: python

        @default_randomizer('pyrado.environments.xy.my', 'MyEnv')
            def create_default_randomizer_my() -> DomainRandomizer:
                <implementation>

    :param env_module: module in which the env class is defined
    :param env_class: environment class name
    :return: decorator for default randomizer provider function
    """

    def register(func):
        DEFAULT_RANDOMIZER_REGISTRY[(env_module, env_class)] = func
        return func

    return register


def create_default_randomizer(env: Union[SimEnv, EnvWrapper]) -> DomainRandomizer:
    """
    Create the default randomizer depending on the passed environment.

    :param env: (wrapped) environment that should be perturbed
    :return: default randomizer
    """
    env_type = type(inner_env(env))

    # Try all env base types. This is more or less equivalent to isinstance
    for cand_type in env_type.__mro__:
        env_module = cand_type.__module__
        env_class = cand_type.__name__
        # Try to get it
        dp = DEFAULT_RANDOMIZER_REGISTRY.get((env_module, env_class))
        if dp:
            return dp()
    else:
        raise pyrado.ValueErr(msg=f"No default randomizer settings for env of type {env_type}!")


def create_conservative_randomizer(env: Union[SimEnv, EnvWrapper]) -> DomainRandomizer:
    """
    Create the default conservative randomizer depending on the passed environment.

    :param env: environment that should be perturbed
    :return: default conservative randomizer
    """
    randomizer = create_default_randomizer(env)
    randomizer.rescale_distr_param("std", 0.5)
    randomizer.rescale_distr_param("cov", 0.5)
    randomizer.rescale_distr_param("halfspan", 0.5)
    return randomizer


def create_zero_var_randomizer(env: Union[SimEnv, EnvWrapper], eps: float = 1e-8) -> DomainRandomizer:
    """
    Create the randomizer which always returns the nominal domain parameter values.

    .. note::
        The variance will not be completely zero as this would lead to invalid distributions (PyTorch checks)

    :param env: environment that should be perturbed
    :param eps: factor to scale the distributions variance with
    :return: randomizer with zero variance for all parameters
    """
    randomizer = create_default_randomizer(env)
    randomizer.rescale_distr_param("std", np.sqrt(eps))
    randomizer.rescale_distr_param("cov", eps)
    randomizer.rescale_distr_param("halfspan", np.sqrt(eps))  # var(U) = 1/12 halspan**2
    return randomizer


def create_empty_randomizer() -> DomainRandomizer:
    """
    Create an empty randomizer independent of the environment to be filled later (using `add_domain_params`).

    :return: empty randomizer
    """
    return DomainRandomizer()


def create_example_randomizer_cata() -> DomainRandomizer:
    """
    Create the randomizer for the `CatapultSim` used for the 'illustrative example' in F. Muratore et al, 2019, TAMPI.

    :return: randomizer based on the nominal domain parameter values
    """
    return DomainRandomizer(
        BernoulliDomainParam(name="planet", mean=None, val_0=0, val_1=1, prob_1=0.7, roundint=True)
    )  # 0 = Mars, 1 = Venus


@default_randomizer("pyrado.environments.one_step.catapult", "CatapultSim")
def create_default_randomizer_cata() -> DomainRandomizer:
    """
    Create the default randomizer for the `CatapultSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.one_step.catapult import CatapultSim

    dp_nom = CatapultSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(
            name="gravity_const", mean=dp_nom["gravity_const"], std=dp_nom["gravity_const"] / 10, clip_lo=1e-3
        ),
        NormalDomainParam(name="stiffness", mean=dp_nom["stiffness"], std=dp_nom["stiffness"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="elongation", mean=dp_nom["elongation"], std=dp_nom["elongation"] / 5, clip_lo=1e-3),
    )


@default_randomizer("pyrado.environments.pysim.ball_on_beam", "BallOnBeamSim")
def create_default_randomizer_bob() -> DomainRandomizer:
    """
    Create the default randomizer for the `BallOnBeamSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim

    dp_nom = BallOnBeamSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(
            name="gravity_const", mean=dp_nom["gravity_const"], std=dp_nom["gravity_const"] / 10, clip_lo=1e-4
        ),
        NormalDomainParam(name="ball_mass", mean=dp_nom["ball_mass"], std=dp_nom["ball_mass"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="ball_radius", mean=dp_nom["ball_radius"], std=dp_nom["ball_radius"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="beam_mass", mean=dp_nom["beam_mass"], std=dp_nom["beam_mass"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="beam_length", mean=dp_nom["beam_length"], std=dp_nom["beam_length"] / 5, clip_lo=1e-3),
        NormalDomainParam(
            name="beam_thickness", mean=dp_nom["beam_thickness"], std=dp_nom["beam_thickness"] / 5, clip_lo=1e-3
        ),
        UniformDomainParam(
            name="friction_coeff", mean=dp_nom["friction_coeff"], halfspan=dp_nom["friction_coeff"], clip_lo=0
        ),
        UniformDomainParam(name="ang_offset", mean=0.0 / 180 * np.pi, halfspan=0.1 / 180 * np.pi),
    )


@default_randomizer("pyrado.environments.pysim.one_mass_oscillator", "OneMassOscillatorSim")
def create_default_randomizer_omo() -> DomainRandomizer:
    """
    Create the default randomizer for the `OneMassOscillatorSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim

    dp_nom = OneMassOscillatorSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name="mass", mean=dp_nom["mass"], std=dp_nom["mass"] / 3, clip_lo=1e-3),
        NormalDomainParam(name="stiffness", mean=dp_nom["stiffness"], std=dp_nom["stiffness"] / 3, clip_lo=1e-3),
        NormalDomainParam(name="damping", mean=dp_nom["damping"], std=dp_nom["damping"] / 3, clip_lo=1e-3),
    )


@default_randomizer("pyrado.environments.pysim.pendulum", "PendulumSim")
def create_default_randomizer_pend() -> DomainRandomizer:
    """
    Create the default randomizer for the `PendulumSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.pendulum import PendulumSim

    dp_nom = PendulumSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(
            name="gravity_const", mean=dp_nom["gravity_const"], std=dp_nom["gravity_const"] / 10, clip_lo=1e-3
        ),
        NormalDomainParam(name="pole_mass", mean=dp_nom["pole_mass"], std=dp_nom["pole_mass"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="pole_length", mean=dp_nom["pole_length"], std=dp_nom["pole_length"] / 10, clip_lo=1e-3),
        NormalDomainParam(
            name="pole_damping", mean=dp_nom["pole_damping"], std=dp_nom["pole_damping"] / 10, clip_lo=1e-3
        ),
        NormalDomainParam(
            name="torque_thold", mean=dp_nom["torque_thold"], std=dp_nom["torque_thold"] / 10, clip_lo=1e-3
        ),
    )


@default_randomizer("pyrado.environments.pysim.quanser_ball_balancer", "QBallBalancerSim")
def create_default_randomizer_qbb() -> DomainRandomizer:
    """
    Create the default randomizer for the `QBallBalancerSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim

    dp_nom = QBallBalancerSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(
            name="gravity_const", mean=dp_nom["gravity_const"], std=dp_nom["gravity_const"] / 10, clip_lo=1e-4
        ),
        NormalDomainParam(name="ball_mass", mean=dp_nom["ball_mass"], std=dp_nom["ball_mass"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="ball_radius", mean=dp_nom["ball_radius"], std=dp_nom["ball_radius"] / 5, clip_lo=1e-3),
        NormalDomainParam(
            name="plate_length", mean=dp_nom["plate_length"], std=dp_nom["plate_length"] / 5, clip_lo=5e-2
        ),
        NormalDomainParam(name="arm_radius", mean=dp_nom["arm_radius"], std=dp_nom["arm_radius"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="gear_ratio", mean=dp_nom["gear_ratio"], std=dp_nom["gear_ratio"] / 4, clip_lo=1e-2),
        NormalDomainParam(
            name="load_inertia", mean=dp_nom["load_inertia"], std=dp_nom["load_inertia"] / 4, clip_lo=1e-6
        ),
        NormalDomainParam(
            name="motor_inertia", mean=dp_nom["motor_inertia"], std=dp_nom["motor_inertia"] / 4, clip_lo=1e-9
        ),
        NormalDomainParam(
            name="motor_back_emf", mean=dp_nom["motor_back_emf"], std=dp_nom["motor_back_emf"] / 4, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="motor_resistance", mean=dp_nom["motor_resistance"], std=dp_nom["motor_resistance"] / 4, clip_lo=1e-4
        ),
        UniformDomainParam(
            name="gear_efficiency",
            mean=dp_nom["gear_efficiency"],
            halfspan=dp_nom["gear_efficiency"] / 4,
            clip_lo=1e-4,
            clip_up=1,
        ),
        UniformDomainParam(
            name="motor_efficiency",
            mean=dp_nom["motor_efficiency"],
            halfspan=dp_nom["motor_efficiency"] / 4,
            clip_lo=1e-4,
            clip_up=1,
        ),
        UniformDomainParam(
            name="combined_damping",
            mean=dp_nom["combined_damping"],
            halfspan=dp_nom["combined_damping"] / 4,
            clip_lo=1e-4,
        ),
        UniformDomainParam(
            name="ball_damping", mean=dp_nom["ball_damping"], halfspan=dp_nom["ball_damping"] / 4, clip_lo=1e-4
        ),
        UniformDomainParam(
            name="voltage_thold_x_pos", mean=dp_nom["voltage_thold_x_pos"], halfspan=dp_nom["voltage_thold_x_pos"] / 3
        ),
        UniformDomainParam(
            name="voltage_thold_x_neg",
            mean=dp_nom["voltage_thold_x_neg"],
            halfspan=abs(dp_nom["voltage_thold_x_neg"]) / 3,
        ),
        UniformDomainParam(
            name="voltage_thold_y_pos", mean=dp_nom["voltage_thold_y_pos"], halfspan=dp_nom["voltage_thold_y_pos"] / 3
        ),
        UniformDomainParam(
            name="voltage_thold_y_neg",
            mean=dp_nom["voltage_thold_y_neg"],
            halfspan=abs(dp_nom["voltage_thold_y_neg"]) / 3,
        ),
        UniformDomainParam(name="offset_th_x", mean=dp_nom["offset_th_x"], halfspan=6.0 / 180 * np.pi),
        UniformDomainParam(name="offset_th_y", mean=dp_nom["offset_th_y"], halfspan=6.0 / 180 * np.pi),
    )


@default_randomizer("pyrado.environments.pysim.quanser_cartpole", "QCartPoleStabSim")
@default_randomizer("pyrado.environments.pysim.quanser_cartpole", "QCartPoleSwingUpSim")
def create_default_randomizer_qcp() -> DomainRandomizer:
    """
    Create the default randomizer for the `QCartPoleSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim

    dp_nom = QCartPoleSim.get_nominal_domain_param(long=False)
    return DomainRandomizer(
        NormalDomainParam(
            name="gravity_const", mean=dp_nom["gravity_const"], std=dp_nom["gravity_const"] / 10, clip_lo=1e-4
        ),
        NormalDomainParam(name="cart_mass", mean=dp_nom["cart_mass"], std=dp_nom["cart_mass"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="pole_mass", mean=dp_nom["pole_mass"], std=dp_nom["pole_mass"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="rail_length", mean=dp_nom["rail_length"], std=dp_nom["rail_length"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="pole_length", mean=dp_nom["pole_length"], std=dp_nom["pole_length"] / 5, clip_lo=1e-2),
        UniformDomainParam(
            name="motor_efficiency",
            mean=dp_nom["motor_efficiency"],
            halfspan=dp_nom["motor_efficiency"] / 4,
            clip_lo=1e-4,
            clip_up=1,
        ),
        UniformDomainParam(
            name="gear_efficiency",
            mean=dp_nom["gear_efficiency"],
            halfspan=dp_nom["gear_efficiency"] / 4,
            clip_lo=1e-4,
            clip_up=1,
        ),
        NormalDomainParam(name="gear_ratio", mean=dp_nom["gear_ratio"], std=dp_nom["gear_ratio"] / 4, clip_lo=1e-4),
        NormalDomainParam(
            name="motor_inertia", mean=dp_nom["motor_inertia"], std=dp_nom["motor_inertia"] / 4, clip_lo=1e-9
        ),
        NormalDomainParam(
            name="pinion_radius", mean=dp_nom["pinion_radius"], std=dp_nom["pinion_radius"] / 5, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="motor_resistance", mean=dp_nom["motor_resistance"], std=dp_nom["motor_resistance"] / 4, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="motor_back_emf", mean=dp_nom["motor_back_emf"], std=dp_nom["motor_back_emf"] / 4, clip_lo=1e-4
        ),
        UniformDomainParam(
            name="combined_damping",
            mean=dp_nom["combined_damping"],
            halfspan=dp_nom["combined_damping"] / 4,
            clip_lo=1e-4,
        ),
        UniformDomainParam(
            name="pole_damping", mean=dp_nom["pole_damping"], halfspan=dp_nom["pole_damping"] / 4, clip_lo=1e-4
        ),
        UniformDomainParam(
            name="cart_friction_coeff",
            mean=dp_nom["cart_friction_coeff"],
            halfspan=dp_nom["cart_friction_coeff"] / 2,
            clip_lo=0,
        ),
    )


@default_randomizer("pyrado.environments.pysim.quanser_qube", "QQubeSwingUpSim")
@default_randomizer("pyrado.environments.pysim.quanser_qube", "QQubeStabSim")
def create_default_randomizer_qq() -> DomainRandomizer:
    """
    Create the default randomizer for the `QQubeSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.quanser_qube import QQubeSim

    dp_nom = QQubeSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(
            name="gravity_const", mean=dp_nom["gravity_const"], std=dp_nom["gravity_const"] / 10, clip_lo=1e-3
        ),
        NormalDomainParam(
            name="motor_resistance", mean=dp_nom["motor_resistance"], std=dp_nom["motor_resistance"] / 5, clip_lo=1e-3
        ),
        NormalDomainParam(
            name="motor_back_emf", mean=dp_nom["motor_back_emf"], std=dp_nom["motor_back_emf"] / 5, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="mass_rot_pole", mean=dp_nom["mass_rot_pole"], std=dp_nom["mass_rot_pole"] / 5, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="length_rot_pole", mean=dp_nom["length_rot_pole"], std=dp_nom["length_rot_pole"] / 5, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="damping_rot_pole", mean=dp_nom["damping_rot_pole"], std=dp_nom["damping_rot_pole"] / 4, clip_lo=1e-9
        ),
        NormalDomainParam(
            name="mass_pend_pole", mean=dp_nom["mass_pend_pole"], std=dp_nom["mass_pend_pole"] / 5, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="length_pend_pole", mean=dp_nom["length_pend_pole"], std=dp_nom["length_pend_pole"] / 5, clip_lo=1e-4
        ),
        NormalDomainParam(
            name="damping_pend_pole",
            mean=dp_nom["damping_pend_pole"],
            std=dp_nom["damping_pend_pole"] / 4,
            clip_lo=1e-9,
        ),
    )


def create_uniform_masses_lengths_randomizer_qq(frac_halfspan: float):
    """
    Get a uniform randomizer that applies to all masses and lengths of the Quanser Qube according to a fraction of their
    nominal parameter values

    :param frac_halfspan: fraction of the nominal parameter value
    :return: `DomainRandomizer` with uniformly distributed masses and lengths
    """
    from pyrado.environments.pysim.quanser_qube import QQubeSim

    dp_nom = QQubeSim.get_nominal_domain_param()
    return DomainRandomizer(
        UniformDomainParam(
            name="mass_pend_pole",
            mean=dp_nom["mass_pend_pole"],
            halfspan=dp_nom["mass_pend_pole"] / frac_halfspan,
            clip_lo=1e-3,
        ),
        UniformDomainParam(
            name="mass_rot_pole",
            mean=dp_nom["mass_rot_pole"],
            halfspan=dp_nom["mass_rot_pole"] / frac_halfspan,
            clip_lo=1e-3,
        ),
        UniformDomainParam(
            name="length_rot_pole",
            mean=dp_nom["length_rot_pole"],
            halfspan=dp_nom["length_rot_pole"] / frac_halfspan,
            clip_lo=1e-2,
        ),
        UniformDomainParam(
            name="length_pend_pole",
            mean=dp_nom["length_pend_pole"],
            halfspan=dp_nom["length_pend_pole"] / frac_halfspan,
            clip_lo=1e-2,
        ),
    )


@default_randomizer("pyrado.environments.rcspysim.ball_on_plate", "BallOnPlateSim")
def create_default_randomizer_bop() -> DomainRandomizer:
    """
    Create the default randomizer for the `BallOnPlateSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.ball_on_plate import BallOnPlateSim

    dp_nom = BallOnPlateSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name="ball_mass", mean=dp_nom["ball_mass"], std=dp_nom["ball_mass"] / 3, clip_lo=1e-2),
        NormalDomainParam(name="ball_radius", mean=dp_nom["ball_radius"], std=dp_nom["ball_radius"] / 3, clip_lo=1e-2),
        NormalDomainParam(name="ball_com_x", mean=dp_nom["ball_com_x"], std=0.003),
        NormalDomainParam(name="ball_com_y", mean=dp_nom["ball_com_y"], std=0.003),
        NormalDomainParam(name="ball_com_z", mean=dp_nom["ball_com_z"], std=0.003),
        UniformDomainParam(
            name="ball_friction_coefficient",
            mean=dp_nom["ball_friction_coefficient"],
            halfspan=dp_nom["ball_friction_coefficient"],
            clip_lo=0,
            clip_hi=1,
        ),
        UniformDomainParam(
            name="ball_rolling_friction_coefficient",
            mean=dp_nom["ball_rolling_friction_coefficient"],
            halfspan=dp_nom["ball_rolling_friction_coefficient"],
            clip_lo=0,
            clip_hi=1,
        ),
        # Vortex only
        UniformDomainParam(name="ball_slip", mean=dp_nom["ball_slip"], halfspan=dp_nom["ball_slip"], clip_lo=0)
        # UniformDomainParam(name='ball_linearvelocitydamnping', mean=0., halfspan=1e-4),
        # UniformDomainParam(name='ball_angularvelocitydamnping', mean=0., halfspan=1e-4)
    )


@default_randomizer("pyrado.environments.rcspysim.planar_insert", "PlanarInsertSim")
def create_default_randomizer_pi() -> DomainRandomizer:
    """
    Create the default randomizer for the `PlanarInsertSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.planar_insert import PlanarInsertSim

    dp_nom = PlanarInsertSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name="link1_mass", mean=dp_nom["link1_mass"], std=dp_nom["link1_mass"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="link2_mass", mean=dp_nom["link2_mass"], std=dp_nom["link2_mass"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="link3_mass", mean=dp_nom["link3_mass"], std=dp_nom["link3_mass"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="link4_mass", mean=dp_nom["link4_mass"], std=dp_nom["link4_mass"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="link5_mass", mean=dp_nom["link4_mass"], std=dp_nom["link4_mass"] / 5, clip_lo=1e-2),
        UniformDomainParam(name="upperwall_pos_offset_z", mean=0, halfspan=0.05, clip_lo=0),  # only increase the gap
    )


@default_randomizer("pyrado.environments.rcspysim.box_shelving", "BoxShelvingPosDSSim")
@default_randomizer("pyrado.environments.rcspysim.box_shelving", "BoxShelvingVelDSSim")
def create_default_randomizer_bs() -> DomainRandomizer:
    """
    Create the default randomizer for the `BoxShelvingSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.box_shelving import BoxShelvingSim

    dp_nom = BoxShelvingSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name="box_length", mean=dp_nom["box_length"], std=dp_nom["box_length"] / 10),
        NormalDomainParam(name="box_width", mean=dp_nom["box_width"], std=dp_nom["box_width"] / 10),
        NormalDomainParam(name="box_mass", mean=dp_nom["box_mass"], std=dp_nom["box_mass"] / 5),
        UniformDomainParam(
            name="box_friction_coefficient",
            mean=dp_nom["box_friction_coefficient"],
            halfspan=dp_nom["box_friction_coefficient"] / 5,
            clip_lo=1e-5,
        ),
    )


@default_randomizer("pyrado.environments.rcspysim.box_lifting", "BoxLiftingPosIKActivationSim")
@default_randomizer("pyrado.environments.rcspysim.box_lifting", "BoxLiftingVelIKActivationSim")
@default_randomizer("pyrado.environments.rcspysim.box_lifting", "BoxLiftingPosDSSim")
@default_randomizer("pyrado.environments.rcspysim.box_lifting", "BoxLiftingVelDSSim")
def create_default_randomizer_bl() -> DomainRandomizer:
    """
    Create the default randomizer for the `BoxLifting`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.box_lifting import BoxLiftingSim

    dp_nom = BoxLiftingSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name="box_length", mean=dp_nom["box_length"], std=dp_nom["box_length"] / 10, clip_lo=5e-2),
        NormalDomainParam(name="box_width", mean=dp_nom["box_width"], std=dp_nom["box_width"] / 10, clip_lo=5e-2),
        NormalDomainParam(name="box_mass", mean=dp_nom["box_mass"], std=dp_nom["box_mass"] / 10),
        UniformDomainParam(
            name="box_friction_coefficient",
            mean=dp_nom["box_friction_coefficient"],
            halfspan=dp_nom["box_friction_coefficient"] / 5,
            clip_lo=1e-3,
        ),
        NormalDomainParam(name="basket_mass", mean=dp_nom["basket_mass"], std=dp_nom["basket_mass"] / 10),
        UniformDomainParam(
            name="basket_friction_coefficient",
            mean=dp_nom["basket_friction_coefficient"],
            halfspan=dp_nom["basket_friction_coefficient"] / 5,
            clip_lo=1e-3,
        ),
    )


@default_randomizer("pyrado.environments.mujoco.openai_half_cheetah", "HalfCheetahSim")
def create_default_randomizer_cth() -> DomainRandomizer:
    from pyrado.environments.mujoco.openai_half_cheetah import HalfCheetahSim

    dp_nom = HalfCheetahSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(
            name="total_mass",
            mean=dp_nom["total_mass"],
            std=dp_nom["total_mass"] / 10,
            clip_lo=1e-3,
        ),
        UniformDomainParam(
            name="tangential_friction_coeff",
            mean=dp_nom["tangential_friction_coeff"],
            halfspan=dp_nom["tangential_friction_coeff"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="torsional_friction_coeff",
            mean=dp_nom["torsional_friction_coeff"],
            halfspan=dp_nom["torsional_friction_coeff"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="rolling_friction_coeff",
            mean=dp_nom["rolling_friction_coeff"],
            halfspan=dp_nom["rolling_friction_coeff"] / 2,
            clip_lo=0.0,
        ),
    )


@default_randomizer("pyrado.environments.mujoco.wam_bic", "WAMJointSpaceCtrlSim")
def create_default_randomizer_wamjsc() -> DomainRandomizer:
    from pyrado.environments.mujoco.wam_jsc import WAMJointSpaceCtrlSim

    dp_nom = WAMJointSpaceCtrlSim.get_nominal_domain_param()
    return DomainRandomizer(
        UniformDomainParam(
            name="joint_1_damping", mean=dp_nom["joint_1_damping"], halfspan=dp_nom["joint_1_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_2_damping", mean=dp_nom["joint_2_damping"], halfspan=dp_nom["joint_2_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_3_damping", mean=dp_nom["joint_3_damping"], halfspan=dp_nom["joint_3_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_4_damping", mean=dp_nom["joint_4_damping"], halfspan=dp_nom["joint_4_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_5_damping", mean=dp_nom["joint_5_damping"], halfspan=dp_nom["joint_5_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_6_damping", mean=dp_nom["joint_6_damping"], halfspan=dp_nom["joint_6_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_7_damping", mean=dp_nom["joint_7_damping"], halfspan=dp_nom["joint_7_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_1_dryfriction",
            mean=dp_nom["joint_1_dryfriction"],
            halfspan=dp_nom["joint_1_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_2_dryfriction",
            mean=dp_nom["joint_2_dryfriction"],
            halfspan=dp_nom["joint_2_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_3_dryfriction",
            mean=dp_nom["joint_3_dryfriction"],
            halfspan=dp_nom["joint_3_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_4_dryfriction",
            mean=dp_nom["joint_4_dryfriction"],
            halfspan=dp_nom["joint_4_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_5_dryfriction",
            mean=dp_nom["joint_5_dryfriction"],
            halfspan=dp_nom["joint_5_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_6_dryfriction",
            mean=dp_nom["joint_6_dryfriction"],
            halfspan=dp_nom["joint_6_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_7_dryfriction",
            mean=dp_nom["joint_7_dryfriction"],
            halfspan=dp_nom["joint_7_dryfriction"] / 2,
            clip_lo=0.0,
        ),
    )


@default_randomizer("pyrado.environments.mujoco.wam_jsc", "WAMJointSpaceCtrlSim")
def create_default_randomizer_wambic() -> DomainRandomizer:
    """
    Create the default randomizer for the MuJoCo-based `WAMBallInCupSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.mujoco.wam_bic import WAMBallInCupSim

    dp_nom = WAMBallInCupSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name="ball_mass", mean=dp_nom["ball_mass"], std=dp_nom["ball_mass"] / 10, clip_lo=1e-2),
        # Ball needs to fit into the cup
        NormalDomainParam(name="cup_scale", mean=dp_nom["cup_scale"], std=dp_nom["cup_scale"] / 5, clip_lo=0.65),
        # Rope won't be more than 3cm off
        NormalDomainParam(
            name="rope_length", mean=dp_nom["rope_length"], std=dp_nom["rope_length"] / 30, clip_lo=0.27, clip_up=0.33
        ),
        UniformDomainParam(
            name="rope_damping", mean=dp_nom["rope_damping"], halfspan=dp_nom["rope_damping"] / 2, clip_lo=1e-6
        ),
        UniformDomainParam(
            name="joint_1_damping", mean=dp_nom["joint_1_damping"], halfspan=dp_nom["joint_1_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_2_damping", mean=dp_nom["joint_2_damping"], halfspan=dp_nom["joint_2_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_3_damping", mean=dp_nom["joint_3_damping"], halfspan=dp_nom["joint_3_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_4_damping", mean=dp_nom["joint_4_damping"], halfspan=dp_nom["joint_4_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_5_damping", mean=dp_nom["joint_5_damping"], halfspan=dp_nom["joint_5_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_6_damping", mean=dp_nom["joint_6_damping"], halfspan=dp_nom["joint_6_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_7_damping", mean=dp_nom["joint_7_damping"], halfspan=dp_nom["joint_7_damping"] / 2, clip_lo=0.0
        ),
        UniformDomainParam(
            name="joint_1_dryfriction",
            mean=dp_nom["joint_1_dryfriction"],
            halfspan=dp_nom["joint_1_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_2_dryfriction",
            mean=dp_nom["joint_2_dryfriction"],
            halfspan=dp_nom["joint_2_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_3_dryfriction",
            mean=dp_nom["joint_3_dryfriction"],
            halfspan=dp_nom["joint_3_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_4_dryfriction",
            mean=dp_nom["joint_4_dryfriction"],
            halfspan=dp_nom["joint_4_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_5_dryfriction",
            mean=dp_nom["joint_5_dryfriction"],
            halfspan=dp_nom["joint_5_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_6_dryfriction",
            mean=dp_nom["joint_6_dryfriction"],
            halfspan=dp_nom["joint_6_dryfriction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_7_dryfriction",
            mean=dp_nom["joint_7_dryfriction"],
            halfspan=dp_nom["joint_7_dryfriction"] / 2,
            clip_lo=0.0,
        ),
    )


@default_randomizer("pyrado.environments.mujoco.openai_ant", "AntSim")
def create_default_randomizer_ant() -> DomainRandomizer:
    """
    Create the default randomizer for the MuJoCo-based `AntSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    return create_default_randomizer_ant_epsilon(0.2)


def create_default_randomizer_ant_epsilon(epsilon: float) -> DomainRandomizer:
    """
    Create a randomizer for the MuJoCo-based `AntSim` whichs domain parameter ranges are controlled by a
    scalar factor.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.mujoco.openai_ant import AntSim

    dp_nom = AntSim.get_nominal_domain_param()
    return DomainRandomizer(
        UniformDomainParam(
            name="wind_x",
            mean=dp_nom["wind_x"],
            halfspan=5 * epsilon,
        ),
        UniformDomainParam(
            name="wind_y",
            mean=dp_nom["wind_y"],
            halfspan=5 * epsilon,
        ),
        UniformDomainParam(
            name="wind_z",
            mean=dp_nom["wind_z"],
            halfspan=5 * epsilon,
        ),
        UniformDomainParam(
            name="gravity",
            mean=dp_nom["gravity"],
            halfspan=0.25 * epsilon * dp_nom["gravity"],
        ),
        UniformDomainParam(
            name="sliding_friction",
            mean=dp_nom["sliding_friction"],
            halfspan=0.3 * epsilon * dp_nom["sliding_friction"],
        ),
        UniformDomainParam(
            name="torsional_friction",
            mean=dp_nom["torsional_friction"],
            halfspan=0.3 * epsilon * dp_nom["torsional_friction"],
        ),
        UniformDomainParam(
            name="rolling_friction",
            mean=dp_nom["rolling_friction"],
            halfspan=0.3 * epsilon * dp_nom["rolling_friction"],
        ),
        UniformDomainParam(
            name="density",
            mean=dp_nom["density"],
            halfspan=0.5 * epsilon * dp_nom["density"],
        ),
    )


@default_randomizer("pyrado.environments.mujoco.openai_humanoid", "HumanoidSim")
def create_default_randomizer_humanoid() -> DomainRandomizer:
    """
    Create the default randomizer for the MuJoCo-based `HumanoidSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    return create_default_randomizer_humanoid_epsilon(0.2)


def create_default_randomizer_humanoid_epsilon(epsilon: float) -> DomainRandomizer:
    """
    Create a randomizer for the MuJoCo-based `HumanoidSim` whichs domain parameter ranges are controlled by a
    scalar factor.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.mujoco.openai_humanoid import HumanoidSim

    dp_nom = HumanoidSim.get_nominal_domain_param()
    return DomainRandomizer(
        UniformDomainParam(
            name="wind_x",
            mean=dp_nom["wind_x"],
            halfspan=5 * epsilon,
        ),
        UniformDomainParam(
            name="wind_y",
            mean=dp_nom["wind_y"],
            halfspan=5 * epsilon,
        ),
        UniformDomainParam(
            name="wind_z",
            mean=dp_nom["wind_z"],
            halfspan=5 * epsilon,
        ),
        UniformDomainParam(
            name="gravity",
            mean=dp_nom["gravity"],
            halfspan=0.25 * epsilon * dp_nom["gravity"],
        ),
        UniformDomainParam(
            name="sliding_friction",
            mean=dp_nom["sliding_friction"],
            halfspan=0.3 * epsilon * dp_nom["sliding_friction"],
        ),
        UniformDomainParam(
            name="torsional_friction",
            mean=dp_nom["torsional_friction"],
            halfspan=0.3 * epsilon * dp_nom["torsional_friction"],
        ),
        UniformDomainParam(
            name="rolling_friction",
            mean=dp_nom["rolling_friction"],
            halfspan=0.3 * epsilon * dp_nom["rolling_friction"],
        ),
        UniformDomainParam(
            name="density",
            mean=dp_nom["density"],
            halfspan=0.5 * epsilon * dp_nom["density"],
        ),
    )


def create_default_domain_param_map_bob() -> Dict[int, Tuple[str, str]]:
    """
    Create the default mapping from indices to domain parameters (as used in the `BayRn` algorithm).

    :return: dict where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("gravity_const", "mean"),
        1: ("gravity_const", "std"),
        2: ("ball_mass", "mean"),
        3: ("ball_mass", "std"),
        4: ("ball_radius", "mean"),
        5: ("ball_radius", "std"),
        6: ("beam_mass", "mean"),
        7: ("beam_mass", "std"),
        8: ("beam_length", "mean"),
        9: ("beam_length", "std"),
        # d_beam ignored
        10: ("friction_coeff", "mean"),
        11: ("friction_coeff", "halfspan"),
        12: ("ang_offset", "mean"),
        13: ("ang_offset", "halfspan"),
    }


def create_default_domain_param_map_omo() -> Dict[int, Tuple[str, str]]:
    """
    Create the default mapping from indices to domain parameters (as used in the `BayRn` algorithm).

    :return: dict where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("mass", "mean"),
        1: ("mass", "std"),
        2: ("stiffness", "mean"),
        3: ("stiffness", "std"),
        5: ("damping", "mean"),
        6: ("damping", "std"),
    }


def create_default_domain_param_map_pend() -> Dict[int, Tuple[str, str]]:
    """
    Create the default mapping from indices to domain parameters (as used in the `BayRn` algorithm).

    :return: dict where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("torque_thold", "mean"),
        1: ("torque_thold", "std"),
    }


def create_default_domain_param_map_qq() -> Dict[int, Tuple[str, str]]:
    """
    Create the default mapping from indices to domain parameters (as used in the `BayRn` algorithm).

    :return: dict where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("mass_pend_pole", "mean"),
        1: ("mass_pend_pole", "std"),
        2: ("mass_rot_pole", "mean"),
        3: ("mass_rot_pole", "std"),
        4: ("length_pend_pole", "mean"),
        5: ("length_pend_pole", "std"),
        6: ("length_rot_pole", "mean"),
        7: ("length_rot_pole", "std"),
        8: ("damping_pend_pole", "mean"),
        9: ("damping_pend_pole", "std"),
        10: ("damping_rot_pole", "mean"),
        11: ("damping_rot_pole", "std"),
    }


def create_default_domain_param_map_wambic() -> Dict[int, Tuple[str, str]]:
    """
    Create the default mapping from indices to domain parameters (as used in the `BayRn` algorithm).

    :return: dict where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("cup_scale", "mean"),
        1: ("cup_scale", "std"),
        2: ("rope_length", "mean"),
        3: ("rope_length", "std"),
        4: ("ball_mass", "mean"),
        5: ("ball_mass", "std"),
        6: ("rope_damping", "mean"),
        7: ("rope_damping", "halfspan"),
        8: ("joint_1_damping", "mean"),
        9: ("joint_1_damping", "halfspan"),
        10: ("joint_2_damping", "mean"),
        11: ("joint_2_damping", "halfspan"),
        12: ("joint_3_damping", "mean"),
        13: ("joint_3_damping", "halfspan"),
        14: ("joint_4_damping", "mean"),
        15: ("joint_4_damping", "halfspan"),
        16: ("joint_5_damping", "mean"),
        17: ("joint_5_damping", "halfspan"),
        18: ("joint_6_damping", "mean"),
        19: ("joint_6_damping", "halfspan"),
        20: ("joint_7_damping", "mean"),
        21: ("joint_7_damping", "halfspan"),
        22: ("joint_1_dryfriction", "mean"),
        23: ("joint_1_dryfriction", "halfspan"),
        24: ("joint_2_dryfriction", "mean"),
        25: ("joint_2_dryfriction", "halfspan"),
        26: ("joint_3_dryfriction", "mean"),
        27: ("joint_3_dryfriction", "halfspan"),
        28: ("joint_4_dryfriction", "mean"),
        29: ("joint_4_dryfriction", "halfspan"),
        30: ("joint_5_dryfriction", "mean"),
        31: ("joint_5_dryfriction", "halfspan"),
        32: ("joint_6_dryfriction", "mean"),
        33: ("joint_6_dryfriction", "halfspan"),
        34: ("joint_7_dryfriction", "mean"),
        35: ("joint_7_dryfriction", "halfspan"),
    }


def create_damping_dryfriction_domain_param_map_wamjsc() -> Dict[int, str]:
    """
    Create a mapping from indices to domain parameters (as used in the `LFI` algorithm).

    :return: dict where the key is the index and the value is the domain parameter name
    """
    return {
        0: "joint_1_damping",
        1: "joint_2_damping",
        2: "joint_3_damping",
        3: "joint_4_damping",
        4: "joint_5_damping",
        5: "joint_6_damping",
        6: "joint_7_damping",
        7: "joint_1_dryfriction",
        8: "joint_2_dryfriction",
        9: "joint_3_dryfriction",
        10: "joint_4_dryfriction",
        11: "joint_5_dryfriction",
        12: "joint_6_dryfriction",
        13: "joint_7_dryfriction",
    }

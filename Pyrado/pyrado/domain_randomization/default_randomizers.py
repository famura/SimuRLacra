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
import numpy as np
from typing import Dict, Tuple, Union

import pyrado
from pyrado.domain_randomization.domain_parameter import BernoulliDomainParam, NormalDomainParam, UniformDomainParam
from pyrado.environments.sim_base import SimEnv
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer


default_randomizer_registry = {}


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
        default_randomizer_registry[(env_module, env_class)] = func
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
        dp = default_randomizer_registry.get((env_module, env_class))
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
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="k", mean=dp_nom["k"], std=dp_nom["k"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="x", mean=dp_nom["x"], std=dp_nom["x"] / 5, clip_lo=1e-3),
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
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-4),
        NormalDomainParam(name="m_ball", mean=dp_nom["m_ball"], std=dp_nom["m_ball"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="r_ball", mean=dp_nom["r_ball"], std=dp_nom["r_ball"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="m_beam", mean=dp_nom["m_beam"], std=dp_nom["m_beam"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="l_beam", mean=dp_nom["l_beam"], std=dp_nom["l_beam"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="d_beam", mean=dp_nom["d_beam"], std=dp_nom["d_beam"] / 5, clip_lo=1e-3),
        UniformDomainParam(name="c_frict", mean=dp_nom["c_frict"], halfspan=dp_nom["c_frict"], clip_lo=0),
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
        NormalDomainParam(name="m", mean=dp_nom["m"], std=dp_nom["m"] / 3, clip_lo=1e-3),
        NormalDomainParam(name="k", mean=dp_nom["k"], std=dp_nom["k"] / 3, clip_lo=1e-3),
        NormalDomainParam(name="d", mean=dp_nom["d"], std=dp_nom["d"] / 3, clip_lo=1e-3),
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
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="m_pole", mean=dp_nom["m_pole"], std=dp_nom["m_pole"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="l_pole", mean=dp_nom["l_pole"], std=dp_nom["l_pole"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="d_pole", mean=dp_nom["d_pole"], std=dp_nom["d_pole"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="tau_max", mean=dp_nom["tau_max"], std=dp_nom["tau_max"] / 10, clip_lo=1e-3),
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
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-4),
        NormalDomainParam(name="m_ball", mean=dp_nom["m_ball"], std=dp_nom["m_ball"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="r_ball", mean=dp_nom["r_ball"], std=dp_nom["r_ball"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="l_plate", mean=dp_nom["l_plate"], std=dp_nom["l_plate"] / 5, clip_lo=5e-2),
        NormalDomainParam(name="r_arm", mean=dp_nom["r_arm"], std=dp_nom["r_arm"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="K_g", mean=dp_nom["K_g"], std=dp_nom["K_g"] / 4, clip_lo=1e-2),
        NormalDomainParam(name="J_l", mean=dp_nom["J_l"], std=dp_nom["J_l"] / 4, clip_lo=1e-6),
        NormalDomainParam(name="J_m", mean=dp_nom["J_m"], std=dp_nom["J_m"] / 4, clip_lo=1e-9),
        NormalDomainParam(name="k_m", mean=dp_nom["k_m"], std=dp_nom["k_m"] / 4, clip_lo=1e-4),
        NormalDomainParam(name="R_m", mean=dp_nom["R_m"], std=dp_nom["R_m"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="eta_g", mean=dp_nom["eta_g"], halfspan=dp_nom["eta_g"] / 4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name="eta_m", mean=dp_nom["eta_m"], halfspan=dp_nom["eta_m"] / 4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name="B_eq", mean=dp_nom["B_eq"], halfspan=dp_nom["B_eq"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="c_frict", mean=dp_nom["c_frict"], halfspan=dp_nom["c_frict"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="V_thold_x_pos", mean=dp_nom["V_thold_x_pos"], halfspan=dp_nom["V_thold_x_pos"] / 3),
        UniformDomainParam(
            name="V_thold_x_neg", mean=dp_nom["V_thold_x_neg"], halfspan=abs(dp_nom["V_thold_x_neg"]) / 3
        ),
        UniformDomainParam(name="V_thold_y_pos", mean=dp_nom["V_thold_y_pos"], halfspan=dp_nom["V_thold_y_pos"] / 3),
        UniformDomainParam(
            name="V_thold_y_neg", mean=dp_nom["V_thold_y_neg"], halfspan=abs(dp_nom["V_thold_y_neg"]) / 3
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
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-4),
        NormalDomainParam(name="m_cart", mean=dp_nom["m_cart"], std=dp_nom["m_cart"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="m_pole", mean=dp_nom["m_pole"], std=dp_nom["m_pole"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="l_rail", mean=dp_nom["l_rail"], std=dp_nom["l_rail"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="l_pole", mean=dp_nom["l_pole"], std=dp_nom["l_pole"] / 5, clip_lo=1e-2),
        UniformDomainParam(name="eta_m", mean=dp_nom["eta_m"], halfspan=dp_nom["eta_m"] / 4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name="eta_g", mean=dp_nom["eta_g"], halfspan=dp_nom["eta_g"] / 4, clip_lo=1e-4, clip_up=1),
        NormalDomainParam(name="K_g", mean=dp_nom["K_g"], std=dp_nom["K_g"] / 4, clip_lo=1e-4),
        NormalDomainParam(name="J_m", mean=dp_nom["J_m"], std=dp_nom["J_m"] / 4, clip_lo=1e-9),
        NormalDomainParam(name="r_mp", mean=dp_nom["r_mp"], std=dp_nom["r_mp"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="R_m", mean=dp_nom["R_m"], std=dp_nom["R_m"] / 4, clip_lo=1e-4),
        NormalDomainParam(name="k_m", mean=dp_nom["k_m"], std=dp_nom["k_m"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="B_eq", mean=dp_nom["B_eq"], halfspan=dp_nom["B_eq"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="B_pole", mean=dp_nom["B_pole"], halfspan=dp_nom["B_pole"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="mu_cart", mean=dp_nom["mu_cart"], halfspan=dp_nom["mu_cart"] / 2, clip_lo=0),
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
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-3),
        NormalDomainParam(name="Rm", mean=dp_nom["Rm"], std=dp_nom["Rm"] / 5, clip_lo=1e-3),
        NormalDomainParam(name="km", mean=dp_nom["km"], std=dp_nom["km"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="Mr", mean=dp_nom["Mr"], std=dp_nom["Mr"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="Lr", mean=dp_nom["Lr"], std=dp_nom["Lr"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="Dr", mean=dp_nom["Dr"], std=dp_nom["Dr"] / 4, clip_lo=1e-9),
        NormalDomainParam(name="Mp", mean=dp_nom["Mp"], std=dp_nom["Mp"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="Lp", mean=dp_nom["Lp"], std=dp_nom["Lp"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="Dp", mean=dp_nom["Dp"], std=dp_nom["Dp"] / 4, clip_lo=1e-9),
    )


def get_uniform_masses_lengths_randomizer_qq(frac_halfspan: float):
    """
    Get a uniform randomizer that applies to all masses and lengths of the Quanser Qube according to a fraction of their
    nominal parameter values

    :param frac_halfspan: fraction of the nominal parameter value
    :return: `DomainRandomizer` with uniformly distributed masses and lengths
    """
    from pyrado.environments.pysim.quanser_qube import QQubeSim

    dp_nom = QQubeSim.get_nominal_domain_param()
    return DomainRandomizer(
        UniformDomainParam(name="Mp", mean=dp_nom["Mp"], halfspan=dp_nom["Mp"] / frac_halfspan, clip_lo=1e-3),
        UniformDomainParam(name="Mr", mean=dp_nom["Mr"], halfspan=dp_nom["Mr"] / frac_halfspan, clip_lo=1e-3),
        UniformDomainParam(name="Lr", mean=dp_nom["Lr"], halfspan=dp_nom["Lr"] / frac_halfspan, clip_lo=1e-2),
        UniformDomainParam(name="Lp", mean=dp_nom["Lp"], halfspan=dp_nom["Lp"] / frac_halfspan, clip_lo=1e-2),
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
            name="joint_1_stiction",
            mean=dp_nom["joint_1_stiction"],
            halfspan=dp_nom["joint_1_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_2_stiction",
            mean=dp_nom["joint_2_stiction"],
            halfspan=dp_nom["joint_2_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_3_stiction",
            mean=dp_nom["joint_3_stiction"],
            halfspan=dp_nom["joint_3_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_4_stiction",
            mean=dp_nom["joint_4_stiction"],
            halfspan=dp_nom["joint_4_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_5_stiction",
            mean=dp_nom["joint_5_stiction"],
            halfspan=dp_nom["joint_5_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_6_stiction",
            mean=dp_nom["joint_6_stiction"],
            halfspan=dp_nom["joint_6_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_7_stiction",
            mean=dp_nom["joint_7_stiction"],
            halfspan=dp_nom["joint_7_stiction"] / 2,
            clip_lo=0.0,
        ),
    )


@default_randomizer("pyrado.environments.mujoco.wam_jsc", "WAMJointSpaceCtrlSim")
def create_default_randomizer_wambic() -> DomainRandomizer:
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
            name="joint_1_stiction",
            mean=dp_nom["joint_1_stiction"],
            halfspan=dp_nom["joint_1_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_2_stiction",
            mean=dp_nom["joint_2_stiction"],
            halfspan=dp_nom["joint_2_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_3_stiction",
            mean=dp_nom["joint_3_stiction"],
            halfspan=dp_nom["joint_3_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_4_stiction",
            mean=dp_nom["joint_4_stiction"],
            halfspan=dp_nom["joint_4_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_5_stiction",
            mean=dp_nom["joint_5_stiction"],
            halfspan=dp_nom["joint_5_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_6_stiction",
            mean=dp_nom["joint_6_stiction"],
            halfspan=dp_nom["joint_6_stiction"] / 2,
            clip_lo=0.0,
        ),
        UniformDomainParam(
            name="joint_7_stiction",
            mean=dp_nom["joint_7_stiction"],
            halfspan=dp_nom["joint_7_stiction"] / 2,
            clip_lo=0.0,
        ),
    )


def get_default_domain_param_map_bob() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters(as for example used in the `BayRn` algorithm).

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("g", "mean"),
        1: ("g", "std"),
        2: ("m_ball", "mean"),
        3: ("m_ball", "std"),
        4: ("r_ball", "mean"),
        5: ("r_ball", "std"),
        6: ("m_beam", "mean"),
        7: ("m_beam", "std"),
        8: ("l_beam", "mean"),
        9: ("l_beam", "std"),
        # d_beam ignored
        10: ("c_frict", "mean"),
        11: ("c_frict", "halfspan"),
        12: ("ang_offset", "mean"),
        13: ("ang_offset", "halfspan"),
    }


def get_default_domain_param_map_omo() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters(as for example used in the `BayRn` algorithm).

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("m", "mean"),
        1: ("m", "std"),
        2: ("k", "mean"),
        3: ("k", "std"),
        5: ("d", "mean"),
        6: ("d", "std"),
    }


def get_default_domain_param_map_pend() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters(as for example used in the `BayRn` algorithm).

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("tau_max", "mean"),
        1: ("tau_max", "std"),
    }


def get_default_domain_param_map_qq() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters(as for example used in the `BayRn` algorithm).

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ("Mp", "mean"),
        1: ("Mp", "std"),
        2: ("Mr", "mean"),
        3: ("Mr", "std"),
        4: ("Lp", "mean"),
        5: ("Lp", "std"),
        6: ("Lr", "mean"),
        7: ("Lr", "std"),
        8: ("Dp", "mean"),
        9: ("Dp", "std"),
        10: ("Dr", "mean"),
        11: ("Dr", "std"),
    }


def get_default_domain_param_map_wambic() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters(as for example used in the `BayRn` algorithm).

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
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
        22: ("joint_1_stiction", "mean"),
        23: ("joint_1_stiction", "halfspan"),
        24: ("joint_2_stiction", "mean"),
        25: ("joint_2_stiction", "halfspan"),
        26: ("joint_3_stiction", "mean"),
        27: ("joint_3_stiction", "halfspan"),
        28: ("joint_4_stiction", "mean"),
        29: ("joint_4_stiction", "halfspan"),
        30: ("joint_5_stiction", "mean"),
        31: ("joint_5_stiction", "halfspan"),
        32: ("joint_6_stiction", "mean"),
        33: ("joint_6_stiction", "halfspan"),
        34: ("joint_7_stiction", "mean"),
        35: ("joint_7_stiction", "halfspan"),
    }

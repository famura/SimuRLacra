/*******************************************************************************
 Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
 Technical University of Darmstadt.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
    or Technical University of Darmstadt, nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
 OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef _DYNAMICALSYSTEM_H_
#define _DYNAMICALSYSTEM_H_

#include "../util/nocopy.h"

#include <Eigen/Core>

#include <vector>

namespace Rcs
{

// forward decl
class PropertySource;

/*! Base class of all Dynamical Systems (DS)
 */
class DynamicalSystem
{
public:
    DynamicalSystem() = default;
    
    virtual ~DynamicalSystem() = default;
    
    // not copy- or movable RCSPYSIM_NOCOPY_NOMOVE(DynamicalSystem)
    
    /**
     * Create a deep copy of this DynamicalSystem.
     * @return deep copy
     */
    virtual DynamicalSystem* clone() const = 0;
    
    // config based factory
    static DynamicalSystem* create(PropertySource* properties, unsigned int innerTaskDim);
    
    /*! Advance the dynamical system one step in time.
     * Compute the velocity x_dot with the desired velocity in state x.
     * For acceleration-based systems, x_dot is prefilled with the current velocity.
     * @param[in,out] x_dot  current velocity, override with new desired velocity
     * @param[in]     x      current state
     * @param[in]     dt     time step size for integration
     */
    virtual void step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const = 0;
    
    /*! Get the goal state for this dynamical system.
     * @return x_des  desired state for the system to pursue
     */
    virtual Eigen::VectorXd getGoal() const = 0;
    
    /*! Set the goal state for this dynamical system.
     * The goal state is the system's desired equilibrium state. In the presence of repellers, this
     * Systems without an explicit goal state do not need to override this. By default, it does othing.
     * @param[in] x_des  desired state for the system to pursue
     */
    virtual void setGoal(const Eigen::VectorXd& x_des) = 0;
    
    /*!
     * Compute the L2 norm between the current state and the goal state.
     * @param[in] x_curr  current system state
     * @return euclidean distance to goal
     */
    virtual double goalDistance(const Eigen::VectorXd& x_curr) const;
    
    unsigned int getStateDim() const;
    
    /*! The desired task space velocity of the DS, which is equal to x_dot coming from step().
     * The robot will most likely not execute the commaded x_dot (e.g. due to the IK or other active DS).
     * We store the desired task space velocity of the DS for potential using it in AMDynamicalSystemActivation or debugging.
     * */
    Eigen::VectorXd x_dot_des;
};


/*! A second-order dynamical system generates desired acceleration values.
 * The acceleration is integrated to output desired velocities.
 */
class DSSecondOrder : public DynamicalSystem
{
public:
    virtual void step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const;
    
    /*! Advance the dynamical system one step in time.
     * Compute the acceleration x_ddot given the velocity x_dot and the state x, and the time step size dt.
     * @param[out] x_ddot fill with desired acceleration, is not initialized
     * @param[in]  x_dot  current velocity
     * @param[in]  x      current state
     * @param[in]  dt     time step size for integration
     */
    virtual void
    step(Eigen::VectorXd& x_ddot, const Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const = 0;
};


/*!
 * Constant Dynamical System
 * When DSConst is used on a velocity Task, a second order system behavior arises.
 */
class DSConst : public DynamicalSystem
{
public:
    /**
     * !Constructor
     * @param[in] constVelocity desired constant velocity of the dynamical system \f$ \dot{x} = c \f$
     */
    DSConst(const Eigen::VectorXd& constVelocity);
    
    virtual DynamicalSystem* clone() const;
    
    void step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const;
    
    Eigen::VectorXd getGoal() const;
    
    void setGoal(const Eigen::VectorXd& x_des);

protected:
    Eigen::VectorXd constVelocity;
};


/*!
 * Linear Dynamical System
 */
class DSLinear : public DynamicalSystem
{
public:
    /**
     * !Constructor
     * @param[in] errorDynamics dynamics matrix
     * @param[in] equilibriumPoint attractor
     */
    DSLinear(const Eigen::MatrixXd& errorDynamics, const Eigen::VectorXd& equilibriumPoint);
    
    virtual DynamicalSystem* clone() const;
    
    void step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const;
    
    Eigen::VectorXd getGoal() const;
    
    void setGoal(const Eigen::VectorXd& x_des);

protected:
    Eigen::MatrixXd errorDynamics;
    Eigen::VectorXd equilibriumPoint;
};


/*!
 * Mass-Spring-Damper Dynamical System
 */
class DSMassSpringDamper : public DSSecondOrder
{
public:
    //! Properties for one spring component.
    struct Spring
    {
        //! Zero position of the spring, force is applied to reach this state.
        Eigen::VectorXd equilibriumPoint;
        //! Stiffness of the spring
        double stiffness;
        
        Spring(Eigen::VectorXd equilibriumPoint, double stiffness) :
            equilibriumPoint(equilibriumPoint), stiffness(stiffness) {}
    };
    
    /*! Constructor
     * @param[in] attractor attractor spring pulling the mass to the goal poisiton (there is only one)
     * @param[in] repellers repeller springs pushing the mass away from points in space
     * @param[in] damping of the dynamical system (there is only on)
     * @param[in] mass mass of the particle (default: unit mass 1kg)
     */
    DSMassSpringDamper(
        const Spring& attractor, const std::vector<Spring>& repellers, const double damping,
        const double mass = 1.0);
    
    virtual DynamicalSystem* clone() const;
    
    void step(Eigen::VectorXd& x_ddot, const Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const;
    
    Eigen::VectorXd getGoal() const;
    
    void setGoal(const Eigen::VectorXd& x_des); // set goal spring

protected:
    //! attractor spring pulling the mass to the goal poisiton (there is only one for every dynamical system)
    Spring attractorSpring;
    //! repeller springs pushing the mass away from points in space (e.g. obstacles)
    std::vector<Spring> repellerSprings;
    //! daming of the dynamical system (there is only one for every dynamical system)
    double damping;
    //! mass of the particle
    double mass;
};


/*!
 * Clampled Nonlinear Mass-Spring-Damper Dynamical System
 */
class DSMassSpringDamperNonlinear : public DSMassSpringDamper
{
public:
    /*! Constructor
     * @param[in] attractor attractor spring pulling the mass to the goal poisiton (there is only one)
     * @param[in] repellers repeller springs pushing the mass away from points in space
     * @param[in] damping of the dynamical system (there is only on)
     * @param[in] mass mass of the particle (default: unit mass 1kg)
     */
    DSMassSpringDamperNonlinear(
        const Spring& attractor, const std::vector<Spring>& repellers, const double damping,
        const double mass = 1.0);
    
    virtual DynamicalSystem* clone() const;
    
    void step(Eigen::VectorXd& x_ddot, const Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const;
};


/*!
 * A dynamical system that only affects a slice of the state x.
 */
class DSSlice : public DynamicalSystem
{
private:
    DynamicalSystem* wrapped;
    unsigned int offset;
    unsigned int length;
    Eigen::VectorXd goal;
public:
    //! constructor
    DSSlice(DynamicalSystem* wrapped, unsigned int offset, unsigned int length);
    
    ~DSSlice();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(DSSlice)
    
    virtual DynamicalSystem* clone() const;
    
    virtual void step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const;
    
    virtual void setGoal(const Eigen::VectorXd& x_des);
    
    virtual Eigen::VectorXd getGoal() const;
    
    virtual double goalDistance(const Eigen::VectorXd& x_cur) const;
    
    DynamicalSystem* getWrapped() const;
};

} /* namespace Rcs */

#endif /* _DYNAMICALSYSTEM_H_ */

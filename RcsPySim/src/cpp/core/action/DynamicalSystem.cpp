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

#include "DynamicalSystem.h"
#include "../config/PropertySource.h"
#include "../util/eigen_matnd.h"

#include <Rcs_macros.h>

#include <memory>
#include <iostream>


namespace Rcs
{

/*! Get an Eigen matrix value from a config entry.
 *
 * Converts the config value to a MatNd, checks the dimensions, and copies the contents into the output reference.
 *
 * @param[out] out           storage for read value.
 * @param[in]  ps            Property source to load from.
 * @param[in]  property      property name
 * @param[in]  expectedRows  Required row count.
 * @param[in]  expectedCols  Required column count.
 *
 * @throws std::invalid_argument if the matrix dimensions do not match.
 */
static bool getMatrixProperty(
    Eigen::MatrixXd& out, PropertySource* ps, const char* property, unsigned int expectedRows,
    unsigned int expectedCols)
{
    MatNd* mat;
    if (!ps->getProperty(mat, property)) {
        return false;
    }
    if (mat->m != expectedRows || mat->n != expectedCols) {
        if (mat->size == expectedRows*expectedCols) {
            // we got a flattened matrix because I have no good idea how to do this in xml
            mat->m = expectedRows;
            mat->n = expectedCols;
        }
        else {
            // dimension mismatch
            MatNd_destroy(mat);
            std::ostringstream os;
            os << "matrix dimension mismatch for " << property;
            os << ": expected " << expectedRows << "x" << expectedCols << " but got " << mat->m << "x" << mat->n;
            throw std::invalid_argument(os.str());
        }
    }
    copyMatNd2Eigen(out, mat);
    MatNd_destroy(mat);
    return true;
}

/*! Get an Eigen vector value from a config entry.
 *
 * Converts the config value to a MatNd, checks the dimensions, and copies the contents into the output reference.
 *
 * @param[out] out           storage for read value.
 * @param[in]  ps            Property source to load from.
 * @param[in]  property      property name
 * @param[in]  expectedSize  Required element count.
 *
 * @throws std::invalid_argument if the matrix dimensions do not match.
 */
static bool getVectorProperty(Eigen::VectorXd& out, PropertySource* ps, const char* property, unsigned int expectedSize)
{
    MatNd* mat;
    if (!ps->getProperty(mat, property)) {
        return false;
    }
    if (mat->m != expectedSize || mat->n != 1) {
        MatNd_destroy(mat);
        std::ostringstream os;
        os << "vector dimension mismatch for " << property;
        os << ": expected " << expectedSize << "(x1)" << " but got " << mat->m << "x" << mat->n;
        throw std::invalid_argument(os.str());
    }
    copyMatNd2Eigen(out, mat);
    MatNd_destroy(mat);
    return true;
}

// The factory
DynamicalSystem* DynamicalSystem::create(PropertySource* properties, unsigned int innerTaskDim)
{
    // Obtain function type and goal
    std::string functionName;
    if (!properties->getProperty(functionName, "function")) {
        throw std::invalid_argument("Missing function specification for the DynamicalSystem!");
    }
    
    // Create a unique pointer for the dynamical systems (will be reset later)
    std::unique_ptr<DynamicalSystem> ds;
    
    if (functionName == "const") {
        Eigen::VectorXd constVelocity;
        if (!getVectorProperty(constVelocity, properties, "DSVelocity", innerTaskDim)) {
            throw std::invalid_argument("Missing constVelocity argument for DSVelocity dynamical system!");
        }
        ds.reset(new DSConst(constVelocity));
    }
    else if (functionName == "lin") {
        Eigen::MatrixXd errorDynamics;
        if (!getMatrixProperty(errorDynamics, properties, "errorDynamics", innerTaskDim, innerTaskDim)) {
            throw std::invalid_argument("Missing errorDynamics argument for DSLinear dynamical system!");
        }
        ds.reset(new DSLinear(errorDynamics, Eigen::VectorXd::Zero(innerTaskDim)));
    }
    else if (functionName == "msd" or functionName == "msd_nlin") {
        double attractorStiffness = 1;
        properties->getProperty(attractorStiffness, "attractorStiffness");
        
        double damping = 0;
        properties->getProperty(damping, "damping");
        
        double mass = 1;
        properties->getProperty(mass, "mass");
        
        // Convert repeller list
        std::vector<DSMassSpringDamper::Spring> repellers;
        auto& repSpec = properties->getChildList("repellers");
        for (auto rep:repSpec) {
            Eigen::VectorXd zp;
            if (!getVectorProperty(zp, rep, "pos", innerTaskDim)) {
                throw std::invalid_argument("Missing position for repeller for DSMassSpringDamper dynamical system!");
            }
            double s = 1;
            rep->getProperty(s, "stiffness");
            repellers.emplace_back(zp, s);
        }
        
        if (functionName == "msd") {
            ds.reset(new DSMassSpringDamper(
                DSMassSpringDamper::Spring(Eigen::VectorXd::Zero(innerTaskDim), attractorStiffness), repellers, damping,
                mass));
        }
        else if (functionName == "msd_nlin") {
            ds.reset(new DSMassSpringDamperNonlinear(
                DSMassSpringDamper::Spring(Eigen::VectorXd::Zero(innerTaskDim), attractorStiffness), repellers, damping,
                mass));
        }
    }
    else {
        std::ostringstream os;
        os << "Unsupported task function name: " << functionName;
        throw std::invalid_argument(os.str());
    }
    
    // Set the goal if specified (common for all DS except DSConst)
    Eigen::VectorXd goal;
    if (getVectorProperty(goal, properties, "goal", innerTaskDim)) {
        ds->setGoal(goal);
    }
    
    return ds.release();
}

double DynamicalSystem::goalDistance(const Eigen::VectorXd& x_curr) const
{
    return (getGoal() - x_curr).norm();  // L2 norm
}

unsigned int DynamicalSystem::getStateDim() const
{
    return x_dot_des.rows();
}


/**
 * DS2ndOrder
 */
void DSSecondOrder::step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const
{
    // Initialize x_ddot as empty matrix
    Eigen::VectorXd x_ddot;
    
    // Compute x_ddot from dynamics
    step(x_ddot, x_dot, x, dt);
    
    // Integrate x_ddot to get x_dot
    x_dot += x_ddot*dt;
}


/**
 * DSConst
 */
DSConst::DSConst(const Eigen::VectorXd& constVelocity) : constVelocity(constVelocity) {}

DynamicalSystem* DSConst::clone() const
{
    return new DSConst(constVelocity);
}

void DSConst::step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const
{
    x_dot = constVelocity;
    REXEC(6) {
        std::cout << "x =" << std::endl << x << std::endl;
        std::cout << "constVelocity (DSConst) =" << std::endl << constVelocity << std::endl;
    }
}

Eigen::VectorXd DSConst::getGoal() const
{
    return constVelocity;
}

void DSConst::setGoal(const Eigen::VectorXd& x_des)
{
    constVelocity = x_des;
}


/**
 * DSLinear
 */
DSLinear::DSLinear(const Eigen::MatrixXd& errorDynamics, const Eigen::VectorXd& equilibriumPoint) :
    errorDynamics(errorDynamics), equilibriumPoint(equilibriumPoint) {}

DynamicalSystem* DSLinear::clone() const
{
    return new DSLinear(errorDynamics, equilibriumPoint);
}

void DSLinear::step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const
{
    x_dot = errorDynamics*(equilibriumPoint - x);
    REXEC(6) {
        std::cout << "x =" << std::endl << x << std::endl;
        std::cout << "diff to goal (DSLinear) =" << std::endl << equilibriumPoint - x << std::endl;
    }
}

Eigen::VectorXd DSLinear::getGoal() const
{
    return equilibriumPoint;
}

void DSLinear::setGoal(const Eigen::VectorXd& x_des)
{
    equilibriumPoint = x_des;
}


/**
 * DSMassSpringDamper
 */
DSMassSpringDamper::DSMassSpringDamper(
    const DSMassSpringDamper::Spring& attractor,
    const std::vector<DSMassSpringDamper::Spring>& repellers,
    const double damping,
    const double mass)
    : attractorSpring(attractor), repellerSprings(repellers), damping(damping), mass(mass) {}

DynamicalSystem* DSMassSpringDamper::clone() const
{
    return new DSMassSpringDamper(attractorSpring, repellerSprings, damping, mass);
}

void DSMassSpringDamper::step(
    Eigen::VectorXd& x_ddot, const Eigen::VectorXd& x_dot, const Eigen::VectorXd& x,
    double dt) const
{
    Eigen::VectorXd sumSprings = Eigen::VectorXd::Zero(x.size()); // sums attractors and repellers
    
    // One attractor
    Eigen::VectorXd diff = attractorSpring.equilibriumPoint - x;
    double dist = diff.norm() + 1e-3;
    double stiffness = attractorSpring.stiffness; // stiffness > 0
    
    sumSprings += stiffness*dist*diff/dist; // force =  stiffness * elongation * normalized_direction
    
    // Iterate over all repellers
    for (auto& spring : repellerSprings) {
        diff = spring.equilibriumPoint - x;
        dist = diff.norm() + 1e-3;
        stiffness = spring.stiffness; // stiffness > 0
        
        sumSprings += -stiffness/dist*diff/dist; // force =  stiffness / elongation * normalized_direction
    }
    
    // Compute acceleration
    x_ddot = (sumSprings - damping*x_dot)/mass;
    
    // Print if debug level is exceeded
    REXEC(6) {
        std::cout << "diff to goal (DSMassSpringDamper) =" << std::endl << diff << std::endl;
        std::cout << "x_ddot (DSMassSpringDamper) =" << std::endl << x_ddot << std::endl;
    }
}

Eigen::VectorXd DSMassSpringDamper::getGoal() const
{
    return attractorSpring.equilibriumPoint;
}

void DSMassSpringDamper::setGoal(const Eigen::VectorXd& x_des)
{
    attractorSpring.equilibriumPoint = x_des;
}


/**
 * DSMassSpringDamperNonlinear
 */
DSMassSpringDamperNonlinear::DSMassSpringDamperNonlinear(
    const DSMassSpringDamper::Spring& attractor,
    const std::vector<DSMassSpringDamperNonlinear::Spring>& repellers,
    const double damping, const double mass)
    : DSMassSpringDamper::DSMassSpringDamper(attractor, repellers, damping, mass) {}

DynamicalSystem* DSMassSpringDamperNonlinear::clone() const
{
    return new DSMassSpringDamperNonlinear(attractorSpring, repellerSprings, damping, mass);
}

void DSMassSpringDamperNonlinear::step(
    Eigen::VectorXd& x_ddot, const Eigen::VectorXd& x_dot, const Eigen::VectorXd& x,
    double dt) const
{
    Eigen::VectorXd sumSprings = Eigen::VectorXd::Zero(x.size()); // sums attractors and repellers
    
    // One attractor
    Eigen::VectorXd diff = attractorSpring.equilibriumPoint - x;
    double dist = fmax(diff.norm(), 5e-2);  // never go below the force that is caused by a 5cm distance
    double stiffness = attractorSpring.stiffness; // stiffness > 0
    
    sumSprings += stiffness*pow(dist, 0.5)*diff/dist; // force =  stiff* sqrt(elong) * norm_dir
    
    // Iterate over all repellers
    for (auto spring : repellerSprings) {
        diff = spring.equilibriumPoint - x;
        dist = fmax(diff.norm(), 5e-2);  // never go below the force that is caused by a 5cm distance
        stiffness = spring.stiffness; // stiffness > 0
        
        sumSprings += stiffness*pow(dist, 0.5)*diff/dist; // force =  stiff * sqrt(elong) * norm_dir
    }
    
    // Compute acceleration
    x_ddot = (sumSprings - damping*x_dot)/mass;
    
    // Print if debug level is exceeded
    REXEC(6) {
        std::cout << "diff to goal (DSMassSpringDamperNonlinear) =" << std::endl << diff << std::endl;
        std::cout << "x_ddot (DSMassSpringDamperNonlinear) =" << std::endl << x_ddot << std::endl;
    }
}


/**
 * DSSlice
 */
DSSlice::DSSlice(DynamicalSystem* wrapped, unsigned int offset, unsigned int length) : wrapped(wrapped), offset(offset),
                                                                                       length(length) {}

DSSlice::~DSSlice()
{
    delete wrapped;
}


DynamicalSystem* DSSlice::clone() const
{
    return new DSSlice(wrapped->clone(), offset, length);
}

void DSSlice::step(Eigen::VectorXd& x_dot, const Eigen::VectorXd& x, double dt) const
{
    //select slice of current x_dot
    Eigen::VectorXd x_dot_s = x_dot.segment(offset, length);
    
    // use slice in wrapped
    wrapped->step(x_dot_s, x.segment(offset, length), dt);
    
    // We want all unselected entries of x_dot to be 0. Since there is no
    // slice complement in eigen, we set the whole vector to 0 and override
    // the selected slice.
    x_dot.setZero(x.size());
    x_dot.segment(offset, length) = x_dot_s;
}

void DSSlice::setGoal(const Eigen::VectorXd& x_des)
{
    goal = x_des;
    wrapped->setGoal(x_des.segment(offset, length));
}


Eigen::VectorXd DSSlice::getGoal() const
{
    return goal;
}

double DSSlice::goalDistance(const Eigen::VectorXd& x_cur) const
{
    return wrapped->goalDistance(x_cur.segment(offset, length));
}

DynamicalSystem* DSSlice::getWrapped() const
{
    return wrapped;
}

} /* namespace Rcs */

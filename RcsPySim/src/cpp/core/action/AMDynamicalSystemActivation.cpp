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

#include "AMDynamicalSystemActivation.h"
#include "ActionModelIK.h"
#include "../util/eigen_matnd.h"

#include <Rcs_macros.h>

#include <utility>


namespace Rcs
{

AMDynamicalSystemActivation::AMDynamicalSystemActivation(
    ActionModel* wrapped, std::vector<DynamicalSystem*> ds, TaskCombinationMethod tcm)
    : ActionModel(wrapped->getGraph()), wrapped(wrapped), dynamicalSystems(std::move(ds)), taskCombinationMethod(tcm)
{
    activation = MatNd_create((unsigned int) dynamicalSystems.size(), 1);
}

AMDynamicalSystemActivation::~AMDynamicalSystemActivation()
{
    delete wrapped;
    delete activation;
    for (auto* ds : dynamicalSystems) {
        delete ds;
    }
}

unsigned int AMDynamicalSystemActivation::getDim() const
{
    return (unsigned int) dynamicalSystems.size();
}

void AMDynamicalSystemActivation::getMinMax(double* min, double* max) const
{
    // All activations are between -1 and 1
    for (unsigned int i = 0; i < getDim(); i++) {
        min[i] = -1;
        max[i] = 1;
    }
}

std::vector<std::string> AMDynamicalSystemActivation::getNames() const
{
    std::vector<std::string> names;
    for (unsigned int i = 0; i < getDim(); ++i) {
        names.push_back("a_" + std::to_string(i));
    }
    return names;
}

void AMDynamicalSystemActivation::computeCommand(
    MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    RCHECK(action->n == 1);  // actions are column vectors
    
    // Remember x_dot from last step as integration input.
    Eigen::VectorXd x_dot_old = x_dot;
    x_dot.setConstant(0);
    
    // Collect data from each DS
    for (unsigned int i = 0; i < getDim(); ++i) {
        // Fill a temp x_dot with the old x_dot
        Eigen::VectorXd x_dot_ds = x_dot_old;
        
        // Step the DS
        dynamicalSystems[i]->step(x_dot_ds, x, dt);
        // Remember x_dot for OMDynamicalSystemDiscrepancy
        dynamicalSystems[i]->x_dot_des = x_dot_ds;
        
        // Combine the individual x_dot of every DS
        switch (taskCombinationMethod) {
            case TaskCombinationMethod::Sum:
            case TaskCombinationMethod::Mean: {
                x_dot += action->ele[i]*x_dot_ds;
                MatNd_set(activation, i, 0, action->ele[i]);
                break;
            }
            
            case TaskCombinationMethod::SoftMax: {
                MatNd* a = NULL;
                MatNd_create2(a, action->m, action->n);
                MatNd_softMax(a, action, action->m);  // action->m is a neat heuristic for beta
                x_dot += MatNd_get(a, i, 0)*x_dot_ds;
                
                MatNd_set(activation, i, 0, MatNd_get(a, i, 0));
                MatNd_destroy(a);
                break;
            }
            
            case TaskCombinationMethod::Product: {
                // Create temp matrix
                MatNd* otherActions = NULL; // other actions are all actions without the current
                MatNd_clone2(otherActions, action);
                MatNd_deleteRow(otherActions, i);
                
                // Treat the actions as probability of events and compute the probability that all other actions are false
                double prod = 1;  // 1 is the neutral element for multiplication
                for (unsigned int idx = 0; idx < otherActions->m; idx++) {
                    REXEC(7) {
                        std::cout << "factor " << (1 - otherActions->ele[idx]) << std::endl;
                    }
                    // One part of the product is always 1
                    prod *= (1 - otherActions->ele[idx]);
                }
                REXEC(7) {
                    std::cout << "prod " << prod << std::endl;
                }
                
                x_dot += action->ele[i]*prod*x_dot_ds;
                MatNd_set(activation, i, 0, action->ele[i]*prod);
                
                MatNd_destroy(otherActions);
                break;
            }
        }
        
        // Print if debug level is exceeded
        REXEC(5) {
            std::cout << "action DS " << i << " = " << action->ele[i] << std::endl;
            std::cout << "x_dot DS " << i << " =" << std::endl << x_dot_ds << std::endl;
        }
    }
    
    if (taskCombinationMethod == TaskCombinationMethod::Mean) {
        
        double normalizer = MatNd_getNormL1(action) + 1e-8;
        x_dot /= normalizer;
        MatNd_constMulSelf(activation, 1./normalizer);
    }
    
    // Integrate to x
    x += x_dot*dt;
    
    // Pass x to wrapped action model
    MatNd x_rcs = viewEigen2MatNd(x);
    
    // Compute the joint angle positions (joint angle velocities, and torques)
    wrapped->computeCommand(q_des, q_dot_des, T_des, &x_rcs, dt);
    
    // Print if debug level is exceeded
    REXEC(5) {
        std::cout << "x_dot (combined MP) =\n" << x_dot << std::endl;
        std::cout << "x (combined MP) =\n" << x << std::endl;
    }
    REXEC(7) {
        MatNd_printComment("q_des", q_des);
        MatNd_printComment("q_dot_des", q_dot_des);
        if (T_des) {
            MatNd_printComment("T_des", T_des);
        }
    }
}

void AMDynamicalSystemActivation::reset()
{
    wrapped->reset();
    // Initialize shapes
    x.setZero(wrapped->getDim());
    x_dot.setZero(wrapped->getDim());
    
    // Obtain current stable action from wrapped action model
    MatNd x_rcs = viewEigen2MatNd(x);
    wrapped->getStableAction(&x_rcs);
}

void AMDynamicalSystemActivation::getStableAction(MatNd* action) const
{
    // All zero activations is stable
    MatNd_setZero(action);
}

Eigen::VectorXd AMDynamicalSystemActivation::getX() const
{
    return x;
}

Eigen::VectorXd AMDynamicalSystemActivation::getXdot() const
{
    return x_dot;
}

ActionModel* AMDynamicalSystemActivation::getWrappedActionModel() const
{
    return wrapped;
}

const std::vector<DynamicalSystem*>& AMDynamicalSystemActivation::getDynamicalSystems() const
{
    return dynamicalSystems;
}

ActionModel* AMDynamicalSystemActivation::clone(RcsGraph* newGraph) const
{
    std::vector<DynamicalSystem*> dsvc;
    for (auto ds : dynamicalSystems) {
        dsvc.push_back(ds->clone());
    }
    
    return new AMDynamicalSystemActivation(wrapped->clone(newGraph), dsvc, taskCombinationMethod);
}

TaskCombinationMethod AMDynamicalSystemActivation::checkTaskCombinationMethod(std::string tcmName)
{
    TaskCombinationMethod tcm;
    if (tcmName == "sum") {
        tcm = TaskCombinationMethod::Sum;
    }
    else if (tcmName == "mean") {
        tcm = TaskCombinationMethod::Mean;
    }
    else if (tcmName == "softmax") {
        tcm = TaskCombinationMethod::SoftMax;
    }
    else if (tcmName == "product") {
        tcm = TaskCombinationMethod::Product;
    }
    else {
        std::ostringstream os;
        os << "Unsupported task combination method: " << tcmName;
        throw std::invalid_argument(os.str());
    }
    return tcm;
}

const char* AMDynamicalSystemActivation::getTaskCombinationMethodName() const
{
    if (taskCombinationMethod == TaskCombinationMethod::Sum) {
        return "sum";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::Mean) {
        return "mean";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::SoftMax) {
        return "softmax";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::Product) {
        return "product";
    }
    else {
        return nullptr;
    }
}

MatNd* AMDynamicalSystemActivation::getActivation() const
{
    return activation;
}

} /* namespace Rcs */


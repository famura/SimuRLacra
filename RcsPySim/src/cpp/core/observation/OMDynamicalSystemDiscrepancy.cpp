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

#include "OMDynamicalSystemDiscrepancy.h"
#include "../action/ActionModelIK.h"
#include "../util/eigen_matnd.h"

#include <ControllerBase.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>

namespace Rcs
{

OMDynamicalSystemDiscrepancy::OMDynamicalSystemDiscrepancy(AMDynamicalSystemActivation* actionModel) : actionModel(
    actionModel)
{
    auto amik = dynamic_cast<AMIKGeneric*>(actionModel->getWrappedActionModel());
    RCHECK_MSG(amik, "AMDynamicalSystemActivation must wrap an AMIKGeneric");
    
    controller = new ControllerBase(actionModel->getGraph());
    for (auto tsk : amik->getController()->getTasks()) {
        controller->add(tsk->clone(actionModel->getGraph()));
    }
    
    x_curr = MatNd_create(controller->getTaskDim(), 1);
}

unsigned int OMDynamicalSystemDiscrepancy::getVelocityDim() const
{
    return 0;  // does not have a velocity field
}

OMDynamicalSystemDiscrepancy::~OMDynamicalSystemDiscrepancy()
{
    delete controller;
    MatNd_destroy(x_curr);
}

unsigned int OMDynamicalSystemDiscrepancy::getStateDim() const
{
    return (unsigned int) controller->getTaskDim(); // equal the number of controller tasks
}

void OMDynamicalSystemDiscrepancy::computeObservation(
    double* state, double* velocity, const MatNd* currentAction,
    double dt) const
{
    // Save last state
    Eigen::VectorXd x_last;
    copyMatNd2Eigen(x_last, x_curr);
    
    // Compute current task state
    controller->computeX(x_curr); // the controller is defined on the actionModel's graph
    
    // Compute actual movement in task space
    Eigen::VectorXd delta_x = viewMatNd2Eigen(x_curr) - x_last;
    
    // Compute discrepancy between what the action model commanded and what the robot did (desired - actual)
    Eigen::VectorXd accumulatedDiscrepancy = actionModel->getX() - viewMatNd2Eigen(x_curr);
    Eigen::VectorXd incrementalDiscrepancy = actionModel->getXdot()*dt - delta_x;
    Eigen::VectorXd velocityDiscrepancy = actionModel->getXdot() - delta_x/dt;
    
    // Print if debug level is exceeded
    REXEC(7) {
        std::cout << "Accumulated dynamical system discrepancy: " << accumulatedDiscrepancy << std::endl;
        std::cout << "Incremental dynamical system discrepancy: " << incrementalDiscrepancy << std::endl;
    }

//    state = &incrementalDiscrepancy(0);
//    velocity = &velocityDiscrepancy(0);
    for (unsigned int i = 0; i < getStateDim(); i++) {
        state[i] = incrementalDiscrepancy[i];
    }
}

void OMDynamicalSystemDiscrepancy::reset()
{
    // fill x_curr with current state
    controller->computeX(x_curr);
}

std::vector<std::string> OMDynamicalSystemDiscrepancy::getStateNames() const
{
    std::vector<std::string> result;
    result.reserve(getStateDim());
    
    // Get names from controller tasks
    for (auto task : controller->getTasks()) {
        std::ostringstream prefix;
        std::string name = "UNSET";
        if (task->getEffector()) {
            // If there is an effector, we use its name
            name = task->getEffector()->name;
        }
        else {
            // If not, e.g. for TaskJoint, then we use
            name = task->getName();
        }
        prefix << "DiscrepDS_" << name << "_";
        for (auto param : task->getParameters()) {
            auto paramName = param.name;
            // The tasks do report their var names, but unfortunately also include the unit in that string.
            // We have to strip that.
            auto spaceIdx = paramName.find(' ');
            if (spaceIdx != std::string::npos) {
                paramName = paramName.substr(0, spaceIdx);
            }
            
            result.push_back(prefix.str() + paramName);
        }
    }
    return result;
}

} /* namespace Rcs */

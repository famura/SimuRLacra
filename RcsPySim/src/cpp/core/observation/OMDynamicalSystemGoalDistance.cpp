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

#include "OMDynamicalSystemGoalDistance.h"
#include "../action/ActionModelIK.h"
#include "../util/eigen_matnd.h"

#include <ControllerBase.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_VecNd.h>

#include <algorithm>
#include <limits>

namespace Rcs
{

OMDynamicalSystemGoalDistance::OMDynamicalSystemGoalDistance(AMDynamicalSystemActivation* actionModel) :
    actionModel(actionModel),
    maxDistance(std::numeric_limits<double>::infinity())
{
    auto amik = dynamic_cast<AMIKGeneric*>(actionModel->getWrappedActionModel());
    RCHECK_MSG(amik, "AMDynamicalSystemActivation must wrap an AMIKGeneric");
    
    controller = new ControllerBase(actionModel->getGraph());
    for (auto tsk : amik->getController()->getTasks()) {
        controller->add(tsk->clone(actionModel->getGraph()));
    }
}

OMDynamicalSystemGoalDistance::~OMDynamicalSystemGoalDistance()
{
    delete controller;
}

unsigned int OMDynamicalSystemGoalDistance::getStateDim() const
{
    return actionModel->getDim();
}

unsigned int OMDynamicalSystemGoalDistance::getVelocityDim() const
{
    return 0;
}

void OMDynamicalSystemGoalDistance::computeObservation(
    double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    // Compute controller state
    Eigen::VectorXd x_curr = Eigen::VectorXd::Zero(controller->getTaskDim());
    MatNd x_curr_mat = viewEigen2MatNd(x_curr);
    controller->computeX(&x_curr_mat);
    
    // Compute goal distance derivative
    auto& tasks = actionModel->getDynamicalSystems();
    for (size_t i = 0; i < tasks.size(); ++i) {
        // Compute distance
        double dist = tasks[i]->goalDistance(x_curr);
        state[i] = dist;
        
        // DEBUG
        REXEC(7) {
            std::cout << "goal distance pos of task " << i << std::endl << state[i] << std::endl;
        }
    }
}

void OMDynamicalSystemGoalDistance::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    VecNd_setZero(minState, getStateDim()); // minimum distance is 0
    VecNd_setElementsTo(maxState, maxDistance, getStateDim());
}

std::vector<std::string> OMDynamicalSystemGoalDistance::getStateNames() const
{
    std::vector<std::string> result;
    result.reserve(getStateDim());
    for (size_t ds = 0; ds < actionModel->getDynamicalSystems().size(); ++ds) {
        std::ostringstream os;
        os << "GD_DS" << ds;
        result.push_back(os.str());
    }
    return result;
}

} /* namespace Rcs */

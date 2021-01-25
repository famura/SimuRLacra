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

#include "RcsPyBot.h"
#include "action/ActionModel.h"
#include "observation/ObservationModel.h"
#include "control/ControlPolicy.h"
//#include "control/MLPPolicy.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_timer.h>
#include <ControllerBase.h>

#include <cmath>

namespace Rcs
{

//namespace {

/*! A simple time-based policy
 * Yields 2 actions: the first one is oscillating round 0 with am amplitude of 0.2 and the second one is constant 0.
 */
class SimpleControlPolicy : public ControlPolicy
{
private:
    //! Internal clock
    double t;

public:
    SimpleControlPolicy() { t = 0; }
    
    virtual void computeAction(MatNd* action, const MatNd* observation)
    {
        action->ele[0] = 0.2*std::cos(2.*M_PI*t)*(135*M_PI/180);
        action->ele[1] = 0.0;
        t += 0.01;
    }
};

//}

RcsPyBot::RcsPyBot(PropertySource* propertySource)
{
    // Load experiment config
    config = ExperimentConfig::create(propertySource);
    
    // Check if all joints are position controlled for skipping a later inverse dynamics control (compliance)
    unsigned int numPosCtrlJoints = 0;
    RCSGRAPH_TRAVERSE_JOINTS(config->graph) {
            if (JNT->jacobiIndex != -1) {
                if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_POSITION) {
                    numPosCtrlJoints++;
                }
            }
        }
    allJointsPosCtrl = config->graph->nJ == numPosCtrlJoints;
    
    // Set MotionControlLayer members
    currentGraph = config->graph;
    desiredGraph = RcsGraph_clone(currentGraph);
    
    // Control policy is set later
    controlPolicy = NULL;
    
    // Init temp matrices, making sure the initial command is identical to the initial state
    q_ctrl = MatNd_clone(desiredGraph->q);
    qd_ctrl = MatNd_clone(desiredGraph->q_dot);
    T_ctrl = MatNd_create(desiredGraph->dof, 1);
    
    observation = config->observationModel->getSpace()->createValueMatrix();
    action = config->actionModel->getSpace()->createValueMatrix();
    
    // ActionModel and observationModel expect a reset() call before they are used
    config->actionModel->reset();
    config->observationModel->reset();
}

RcsPyBot::~RcsPyBot()
{
    // Delete temporary matrices
    MatNd_destroy(q_ctrl);
    MatNd_destroy(qd_ctrl);
    MatNd_destroy(T_ctrl);
    
    MatNd_destroy(observation);
    MatNd_destroy(action);
    
    // The hardware components also use currentGraph, so it may only be destroyed by the MotionControlLayer destructor.
    // however, currentGraph is identical to config->graph, which is owned.
    // to solve this, set config->graph to NULL.
    config->graph = NULL;
    // Also, desiredGraph is a clone, so it must be destroyed. Can't set MotionControlLayer::ownsDesiredGraph = true
    // since it's private, so do it manually here.
    RcsGraph_destroy(desiredGraph);
    
    // Delete experiment config
    delete config;
}

void RcsPyBot::setControlPolicy(ControlPolicy* controlPolicy)
{
    std::unique_lock<std::mutex> lock(controlPolicyMutex);
    this->controlPolicy = controlPolicy;
    if (controlPolicy == NULL) {
        // Command initial state
        RcsGraph_getDefaultState(desiredGraph, q_ctrl);
        MatNd_setZero(qd_ctrl);
        MatNd_setZero(T_ctrl);
    }
    // Reset model states
    config->observationModel->reset();
    config->actionModel->reset();
}

void RcsPyBot::updateControl()
{
    // Aggressive locking here is ok, setControlPolicy doesn't take long
    std::unique_lock<std::mutex> lock(controlPolicyMutex);
    
    // Read observation from current graph
    config->observationModel->computeObservation(observation, action, config->dt);
    
    // Compute action
    if (controlPolicy != NULL) {
        controlPolicy->computeAction(action, observation);
        
        // Run action through action model
        config->actionModel->computeCommand(q_ctrl, qd_ctrl, T_ctrl, action, getCallbackUpdatePeriod());
    }
    
    // Inverse dynamics in joint space (compliance control)
    if (!allJointsPosCtrl) {
        Rcs::ControllerBase::computeInvDynJointSpace(T_ctrl, config->graph, q_ctrl, 1000.);
    }
    
    // Update desired state graph
    
    // TODO if(writeCommands), but seriously, shouldn't the component handle that stuff itself?
    // Command action to hardware (and update desiredGraph)
    setMotorCommand(q_ctrl, qd_ctrl, T_ctrl);
    
    // Can unlock now, lock only guards controlPolicy and ctrl vectors
    lock.unlock();
    
    // Log data
    double reward = 0.0; // so far, the reward is only computed on the python side
    logger.record(observation, action, reward);
}

MatNd* RcsPyBot::getObservation() const
{
    return observation;
}

MatNd* RcsPyBot::getAction() const
{
    return action;
}


} /* namespace Rcs */

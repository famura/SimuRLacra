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

#include "PolicyComponent.h"
#include "action/ActionModel.h"
#include "action/ActionModelIK.h"
#include "action/AMDynamicalSystemActivation.h"
#include "action/AMJointControlPosition.h"
#include "observation/OMCollisionCost.h"
#include "util/eigen_matnd.h"
#include "util/string_format.h"

#include <EntityBase.h>

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_cmdLine.h>

#include <memory>

Rcs::PolicyComponent::PolicyComponent(
    Rcs::EntityBase* entity,
    Rcs::PropertySource* settings,
    bool computeJointVelocities) :
    ComponentBase(entity), computeJointVelocities(computeJointVelocities)
{
    // Load experiment
    experiment = ExperimentConfig::create(settings);
    
    // The dt value from the settings should be propagated to the whole entity
    entity->setDt(experiment->dt);
    
    // Try to load policy from command line args
    CmdLineParser argP;
    std::string policyFile;
    if (argP.getArgument("-policy", &policyFile, "Policy file to use")) {
        std::string policyType;
        if (!argP.getArgument("-policyType", &policyType, "Type of policy to load")) {
            // Try to determine from file
            if (policyFile.size() >= 4 && policyFile.substr(policyFile.size() - 4) == ".pth") {
                policyType = "torch";
            }
            else {
                RFATAL("Cannot determine policy type from policy file %s.", policyFile.c_str());
            }
        }
        RCHECK(!policyType.empty());
        policy = Rcs::ControlPolicy::create(policyType.c_str(), policyFile.c_str());
    }
    else {
        // Try to load from config
        auto policyConfig = settings->getChild("policy");
        if (policyConfig->exists()) {
            // Load from config
            policy = Rcs::ControlPolicy::create(policyConfig);
        }
        else {
            RFATAL("No policy");
        }
    }
    
    // Start with inactive policy
    policyActive = false;
    eStop = false;
    eRec = false;
    goHome = false;
    renderingInitialized = false;
    
    // Init temp matrices, making sure the initial command is identical to the initial state
    observation = experiment->observationModel->getSpace()->createValueMatrix();
    action = experiment->actionModel->getSpace()->createValueMatrix();
    
    desiredGraph = RcsGraph_clone(experiment->graph);
    
    T_ctrl = MatNd_create(experiment->graph->dof, 1);
    
    // Copy collision model for desired graph
    collisionMdl = RcsCollisionModel_clone(experiment->collisionMdl, desiredGraph);
    
    // Setup go home policy
    std::unique_ptr<ActionModel> goHomeWrappedAM;
    auto existingAMIK = experiment->actionModel->unwrap<ActionModelIK>();
    if (existingAMIK) {
        // Use existing IK definition
        goHomeWrappedAM.reset(existingAMIK->clone(experiment->graph));
    }
    else {
        // No predefined IK, use joint position control
        RLOG(0, "Couldn't find IK definition in action model, using joint position control for go home.");
        goHomeWrappedAM.reset(new AMJointControlPosition(experiment->graph));
    }
    goHomeWrappedAM->reset();
    
    // Extract home position
    Eigen::VectorXd homePos;
    homePos.setZero(goHomeWrappedAM->getDim());
    MatNd hpMat = viewEigen2MatNd(homePos);
    goHomeWrappedAM->getStableAction(&hpMat);
    
    // Setup go home DS
    PropertySource* ghConfig = experiment->properties->getChild("goHomeDS");
    if (ghConfig->exists()) {
        // Read from config, but always set goal pos to home pos
        goHomeDS = DynamicalSystem::create(ghConfig, goHomeWrappedAM->getDim());
        goHomeDS->setGoal(homePos);
    }
    else {
        // Create a simple linear ds
        goHomeDS = new DSLinear(
            Eigen::MatrixXd::Identity(goHomeWrappedAM->getDim(), goHomeWrappedAM->getDim()), homePos);
    }
    goHomeAM = new AMDynamicalSystemActivation(goHomeWrappedAM.release(), {goHomeDS});
    
    // Register subscribers
    subscribe();
    
    // Reset models and policy
    experiment->observationModel->reset();
    experiment->actionModel->reset();
    policy->reset();
    goHomeAM->reset();
}

Rcs::PolicyComponent::~PolicyComponent()
{
    unsubscribe();
    
    RcsCollisionModel_destroy(collisionMdl);
    
    delete goHomeAM;
    // goHomeDS is owned by goHomeAM, must not delete separately
    
    // Delete temporary matrices
    MatNd_destroy(T_ctrl);
    RcsGraph_destroy(desiredGraph);
    
    MatNd_destroy(observation);
    MatNd_destroy(action);
    
    delete policy;
    delete experiment;
}

void Rcs::PolicyComponent::subscribe()
{
#define SUBSCRIBE(name) ComponentBase::subscribe(#name, &PolicyComponent::on##name)
    SUBSCRIBE(UpdatePolicy);
    SUBSCRIBE(InitFromState);
    SUBSCRIBE(EmergencyStop);
    SUBSCRIBE(EmergencyRecover);
    SUBSCRIBE(Render);
    SUBSCRIBE(Print);
    SUBSCRIBE(PolicyStart);
    SUBSCRIBE(PolicyPause);
    SUBSCRIBE(PolicyReset);
    SUBSCRIBE(GoHome);
}

void Rcs::PolicyComponent::onUpdatePolicy(const RcsGraph* state)
{
    if (eStop) {
        return;
    }
    
    // Update current graph
    if (computeJointVelocities) {
        if (eRec) {
            // Don't compute velocity on emergency recovery to avoid instabilities
            MatNd_setZero(experiment->graph->q_dot);
        }
        else {
            // Compute velocity using finite differences.
            // The last state is still stored in the experiment's graph, since we only update that later
            MatNd_sub(experiment->graph->q_dot, state->q, experiment->graph->q);
            MatNd_constMulSelf(experiment->graph->q_dot, 1/experiment->dt);
        }
    }
    else {
        // Use observed velocities
        MatNd_copy(experiment->graph->q_dot, state->q_dot);
    }
    
    // Update graph
    MatNd_copy(experiment->graph->q, state->q);
    RcsGraph_computeForwardKinematics(experiment->graph, NULL, NULL);
    
    // Reset observation and action model on recovery
    if (eRec) {
        experiment->observationModel->reset();
        experiment->actionModel->reset();
        goHomeAM->reset();
        // Don't reset policy since we might want it to continue.
        // Reset desired graph to current graph to avoid sending commands right away
        RcsGraph_setState(desiredGraph, experiment->graph->q, experiment->graph->q_dot);
    }
    
    // Compute observation
    experiment->observationModel->computeObservation(observation, action, experiment->dt);
    
    // Compute action
    if (policyActive) {
        policy->computeAction(action, observation);
        
        // Run action through action model
        experiment->actionModel->computeCommand(desiredGraph->q, desiredGraph->q_dot, T_ctrl, action, experiment->dt);
        // Update desired graph
        RcsGraph_computeForwardKinematics(desiredGraph, NULL, NULL);
    }
    else if (goHome) {
        // Invoke goHome MP
        double goHomeAct = 1;
        MatNd ghaMat = MatNd_fromPtr(1, 1, &goHomeAct);
        goHomeAM->computeCommand(desiredGraph->q, desiredGraph->q_dot, T_ctrl, &ghaMat, experiment->dt);
        // Update desired graph
        RcsGraph_computeForwardKinematics(desiredGraph, NULL, NULL);
    }
    
    // Joint limit check
    if (this->jointLimitCheck) {
        unsigned int aor = RcsGraph_numJointLimitsViolated(desiredGraph, true);
        if (aor > 0) {
            RLOG(0, "%d joint limit violations - triggering emergency stop", aor);
            getEntity()->publish("EmergencyStop");
            RcsGraph_printState(desiredGraph, desiredGraph->q);
        }
    }
    
    // Collision check
    if (this->collisionCheck && collisionMdl) {
        // Compute collisions
        RcsCollisionModel_compute(collisionMdl);
        // Check distance
        const double distLimit = 0.001;
        double minDist = RcsCollisionMdl_getMinDist(collisionMdl);
        if (minDist < distLimit) {
            RLOG(0, "Found collision distance of %f (must be >%f) - triggering emergency stop",
                 minDist, distLimit);
            RcsCollisionModel_fprintCollisions(stdout, collisionMdl, distLimit);
            getEntity()->publish("EmergencyStop");
        }
    }
    
    // Wherever we are, emergency recovery is done
    eRec = false;
}

void Rcs::PolicyComponent::onInitFromState(const RcsGraph* target)
{
    RLOG(0, "PolicyComponent::onInitFromState()");
    // Update current graph
    MatNd_copy(experiment->graph->q, target->q);
    if (computeJointVelocities) {
        // Set zero velocities on reset
        MatNd_setZero(experiment->graph->q_dot);
    }
    else {
        // Use observed velocities
        MatNd_copy(experiment->graph->q_dot, target->q_dot);
    }
    RcsGraph_computeForwardKinematics(experiment->graph, NULL, NULL);
    
    // Reset action and activation model to enforce consistent state.
    experiment->observationModel->reset();
    experiment->actionModel->reset();
    
    // Update initial desired states
    RcsGraph_setState(desiredGraph, experiment->graph->q, experiment->graph->q_dot);
    MatNd_setZero(T_ctrl);
}

void Rcs::PolicyComponent::onEmergencyStop()
{
    eStop = true;
    // Make sure that the policy is inactive
    policyActive = false;
    goHome = false;
}

void Rcs::PolicyComponent::onEmergencyRecover()
{
    if (!eStop) {
        RLOG(0, "Not in emergency stop, ignored.");
        return;
    }
    eStop = false;
    // Mark recovery for next update call
    eRec = true;
}

void Rcs::PolicyComponent::onPolicyStart()
{
    if (eStop) {
        RLOG(0, "Emergency stop is active, cannot start policy.");
        return;
    }
    goHome = false;
    policyActive = true;
    
    // Reset action and observation model to make sure they're aware of the current position
    experiment->observationModel->reset();
    experiment->actionModel->reset();
    
    // Does not reset policy, to do that use explicit resetPolicy()
}

void Rcs::PolicyComponent::onPolicyPause()
{
    policyActive = false;
    goHome = false;
}

void Rcs::PolicyComponent::onPolicyReset()
{
    policy->reset();
}

void Rcs::PolicyComponent::onGoHome()
{
    if (eStop) {
        RLOG(0, "Emergency stop is active, cannot start go home policy.");
        return;
    }
    // Deactivate policy
    policyActive = false;
    
    // Activate go home policy
    goHome = true;
    
    // Reset go home policy to make sure it's aware of the current position
    goHomeAM->reset();
}

void Rcs::PolicyComponent::onRender()
{
    // Publish the IK graph
    getEntity()->publish<std::string, const RcsGraph*>("RenderGraph", "IK", desiredGraph);
    if (collisionMdl) {
        getEntity()->publish<const MatNd*>("RenderLines", collisionMdl->cp);
    }
    
    // Perform rendering initialization on first render call
    if (!this->renderingInitialized) {
        getEntity()->publish<std::string, std::string>("RenderCommand", "IK", "setGhostMode");
        
        this->renderingInitialized = true;
    }
}

std::string Rcs::PolicyComponent::getStateText() const
{
    if (eStop) {
        return "PolicyComponent: emergency stop!";
    }
    else if (policyActive) {
        return "PolicyComponent: policy active";
    }
    else if (goHome) {
        return "PolicyComponent: go home";
    }
    else {
        return "PolicyComponent: inactive";
    }
}

void Rcs::PolicyComponent::onPrint()
{
    printf("Desired state from action model:%n");
    RcsGraph_fprintModelState(stdout, desiredGraph, desiredGraph->q);
}

Rcs::ExperimentConfig* Rcs::PolicyComponent::getExperiment() const
{
    return experiment;
}

Rcs::ControlPolicy* Rcs::PolicyComponent::getPolicy() const
{
    return policy;
}

const MatNd* Rcs::PolicyComponent::getObservation() const
{
    return observation;
}

const MatNd* Rcs::PolicyComponent::getAction() const
{
    return action;
}

const MatNd* Rcs::PolicyComponent::getJointCommandPtr() const
{
    return desiredGraph->q;
}

RcsGraph* Rcs::PolicyComponent::getDesiredGraph() const
{
    return desiredGraph;
}

void Rcs::PolicyComponent::setJointLimitCheck(bool jointLimitCheck)
{
    PolicyComponent::jointLimitCheck = jointLimitCheck;
}

void Rcs::PolicyComponent::setCollisionCheck(bool collisionCheck)
{
    PolicyComponent::collisionCheck = collisionCheck;
}

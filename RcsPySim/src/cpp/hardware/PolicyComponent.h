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

#ifndef RCSPYSIM_POLICYCOMPONENT_H
#define RCSPYSIM_POLICYCOMPONENT_H

#include <config/PropertySource.h>
#include <ExperimentConfig.h>
#include <control/ControlPolicy.h>

#include <ComponentBase.h>
#include <Rcs_MatNd.h>

namespace Rcs
{

class DynamicalSystem;

/**
 * Wraps RcsPySim Experiment and ControlPolicy for use in ECS.
 */
class PolicyComponent : public ComponentBase
{
public:
    PolicyComponent(EntityBase* entity, PropertySource* settings, bool computeJointVelocities = false);
    
    virtual ~PolicyComponent();
    
    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(PolicyComponent)
    
    ExperimentConfig* getExperiment() const;
    
    ControlPolicy* getPolicy() const;
    
    const MatNd* getObservation() const;
    
    const MatNd* getAction() const;
    
    const MatNd* getJointCommandPtr() const;
    
    RcsGraph* getDesiredGraph() const;
    
    // get a text describing the current state of the command fsm
    std::string getStateText() const;
    
    void setJointLimitCheck(bool jointLimitCheck);
    
    void setCollisionCheck(bool collisionCheck);

private:
    // event handlers
    void subscribe();

//    void onStart();
//    void onStop();
    void onUpdatePolicy(const RcsGraph* state);
    
    void onInitFromState(const RcsGraph* target);
    
    void onEmergencyStop();
    
    void onEmergencyRecover();
    
    void onRender();
    
    void onPrint();
    
    void onPolicyStart();
    
    void onPolicyPause();
    
    void onPolicyReset();
    
    void onGoHome();
    
    
    // experiment to use
    ExperimentConfig* experiment;
    // policy to use
    ControlPolicy* policy;
    
    // the robot doesn't provide joint velocities. to work around that, compute them using finite differences
    bool computeJointVelocities;
    
    // true to check joint limits
    bool jointLimitCheck;
    // true to check for collisions
    bool collisionCheck;
    
    // collision model for collision check
    RcsCollisionMdl* collisionMdl;
    
    // true if the policy should be active
    bool policyActive;
    // true while in emergency stop
    bool eStop;
    // true in the first onUpdatePolicy call after onEmergencyRecover
    bool eRec;
    // allows to catch the first render call
    bool renderingInitialized;
    
    
    // Temporary matrices
    MatNd* observation;
    MatNd* action;
    
    // graph containing the desired states
    RcsGraph* desiredGraph;
    
    // go home policy
    bool goHome;
    DynamicalSystem* goHomeDS;
    ActionModel* goHomeAM;
    
    // dummy, might be filled but is not used
    MatNd* T_ctrl;
    
};

}

#endif //RCSPYSIM_POLICYCOMPONENT_H

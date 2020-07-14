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

#include "ExperimentConfig.h"
#include "action/ActionModelIK.h"
#include "action/AMDynamicalSystemActivation.h"
#include "observation/OMCombined.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDSphereRadius.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_macros.h>
#include <TaskPosition3D.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <memory>

namespace Rcs
{

class ECTargetTracking : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* left = RcsGraph_getBodyByName(graph, "PowerGrasp_L");
        RCHECK(left);
        RcsBody* right = RcsGraph_getBodyByName(graph, "PowerGrasp_R");
        RCHECK(right);
        
        // Control effector positions (not orientation)
        std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
        innerAM->addTask(new TaskPosition3D(graph, left, nullptr, nullptr));
        innerAM->addTask(new TaskPosition3D(graph, right, nullptr, nullptr));
        
        // Incorporate collision costs into IK
        if (properties->getPropertyBool("collisionAvoidanceIK", true)) {
            REXEC(4) {
                std::cout << "IK considers the provided collision model" << std::endl;
            }
            innerAM->setupCollisionModel(collisionMdl);
        }
        
        // Obtain task data
        std::vector<std::unique_ptr<DynamicalSystem>> tasks;
        auto& tsLeft = properties->getChildList("tasksLeft");
        for (auto tsk : tsLeft) {
            DynamicalSystem* ds = DynamicalSystem::create(tsk, 3);
            tasks.emplace_back(new DSSlice(ds, 0, 3));
        }
        auto& tsRight = properties->getChildList("tasksRight");
        for (auto tsk : tsRight) {
            DynamicalSystem* ds = DynamicalSystem::create(tsk, 3);
            tasks.emplace_back(new DSSlice(ds, 3, 3));
        }
        if (tasks.empty()) {
            throw std::invalid_argument("No tasks specified!");
        }
        
        // Setup task-based action model
        std::vector<DynamicalSystem*> taskRel;
        for (auto& task : tasks) {
            taskRel.push_back(task.release());
        }
        
        // Get the method how to combine the movement primitives / tasks given their activation
        std::string taskCombinationMethod = "mean";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(taskCombinationMethod);
        
        return new AMDynamicalSystemActivation(innerAM.release(), taskRel, tcm);
    }
    
    virtual ObservationModel* createObservationModel()
    {
        // Observe effector positions
        std::unique_ptr<OMCombined> fullState(new OMCombined());
        
        auto left = new OMBodyStateLinear(graph, "PowerGrasp_L");
        fullState->addPart(left);
        
        auto right = new OMBodyStateLinear(graph, "PowerGrasp_R");
        fullState->addPart(right);
        
        auto amAct = actionModel->unwrap<AMDynamicalSystemActivation>();
        RCHECK(amAct);
        fullState->addPart(new OMDynamicalSystemGoalDistance(amAct));
        
        if (properties->getPropertyBool("observeCollisionCost", true) & (collisionMdl != nullptr)) {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        if (properties->getPropertyBool("observePredictedCollisionCost", false) & (collisionMdl != nullptr)) {
            // Get the horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add the collision cost observation model
            auto omCollPred = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, horizon);
            fullState->addPart(omCollPred);
        }
        
        
        return fullState.release();
    }

public:
    void
    getHUDText(
        std::vector<std::string>& linesOut, double currentTime, const MatNd* obs, const MatNd* currentAction,
        PhysicsBase* simulator, PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != nullptr) {
            simName = simulator->getClassName();
        }
        
        linesOut.emplace_back(string_format(
            "physics engine: %s        simulation time:             %2.3f s",
            simName, currentTime));
        
        unsigned int numPosCtrlJoints = 0;
        unsigned int numTrqCtrlJoints = 0;
        // Iterate over unconstrained joints
        RCSGRAPH_TRAVERSE_JOINTS(graph) {
                if (JNT->jacobiIndex != -1) {
                    if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_POSITION) {
                        numPosCtrlJoints++;
                    }
                    else if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_TORQUE) {
                        numTrqCtrlJoints++;
                    }
                }
            }
        linesOut.emplace_back(
            string_format("num joints:    %d total, %d pos ctrl, %d trq ctrl", graph->nJ, numPosCtrlJoints,
                          numTrqCtrlJoints));
        
        linesOut.emplace_back(string_format(
            " left hand pg:     [% 3.2f,% 3.2f,% 3.2f] m",
            obs->ele[0], obs->ele[1], obs->ele[2]));
        
        linesOut.emplace_back(string_format(
            "right hand pg:     [% 3.2f,% 3.2f,% 3.2f] m",
            obs->ele[3], obs->ele[4], obs->ele[5]));
        
        linesOut.emplace_back(string_format(
            "goal distance:     [% 3.2f,% 3.2f] m",
            obs->ele[6], obs->ele[7]));
        
        auto omColl = observationModel->findOffsets<OMCollisionCost>();
        if (omColl) {
            linesOut.emplace_back(string_format(
                "collision cost:        % 3.2f",
                obs->ele[omColl.pos]));
        }
        
        auto omCollPred = observationModel->findOffsets<OMCollisionCostPrediction>();
        if (omCollPred) {
            linesOut.emplace_back(string_format(
                "collision cost (pred): % 3.2f",
                obs->ele[omCollPred.pos]));
        }
    }
    
};

// Register
static ExperimentConfigRegistration<ECTargetTracking> RegTargetTracking("TargetTracking");

}

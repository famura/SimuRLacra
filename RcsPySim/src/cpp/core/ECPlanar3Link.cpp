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
#include "action/AMIntegrate1stOrder.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMJointControlPosition.h"
#include "action/AMDynamicalSystemActivation.h"
#include "action/AMIKControllerActivation.h"
#include "initState/ISSPlanar3Link.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "observation/OMPartial.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <TaskPosition1D.h>
#include <TaskVelocity1D.h>
#include <TaskDistance.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Rcs
{

class ECPlanar3Link : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "joint_pos";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Common for the action models
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
        
        if (actionModelType == "joint_pos") {
            return new AMJointControlPosition(graph);
        }
        else if (actionModelType == "joint_vel") {
            double max_action = 90*M_PI/180; // [rad/s]
            return new AMIntegrate1stOrder(new AMJointControlPosition(graph), max_action);
        }
        else if (actionModelType == "joint_acc") {
            double max_action = 120*M_PI/180; // [rad/s^2]
            return new AMIntegrate2ndOrder(new AMJointControlPosition(graph), max_action);
        }
        else if (actionModelType == "ik_activation") {
            // Create the action model
            auto amIK = new AMIKGeneric(graph);
            std::vector<TaskGenericIK*> tasks;
            
            // Check if the tasks are defined on position or task level. Adapt their parameters if desired.
            if (properties->getPropertyBool("positionTasks", true)) {
                
                // Get the method how to combine the movement primitives / tasks given their activation
                std::string taskCombinationMethod = "mean";
                properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
                TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(
                    taskCombinationMethod);
                
                // Override the action model
                amIK = new AMIKControllerActivation(graph, tcm);
                
                RcsBody* goal1 = RcsGraph_getBodyByName(graph, "Goal1");
                RcsBody* goal2 = RcsGraph_getBodyByName(graph, "Goal2");
                RcsBody* goal3 = RcsGraph_getBodyByName(graph, "Goal3");
                RCHECK(goal1);
                RCHECK(goal2);
                RCHECK(goal3);
                int i = 0;

//                tasks.emplace_back(new TaskDistance(graph, effector, goal1));
//                tasks.emplace_back(new TaskDistance(graph, effector, goal2));
//                tasks.emplace_back(new TaskDistance(graph, effector, goal3));
//                for (auto task : tasks) {
//                    std::stringstream taskName;
//                    taskName << "Distance " << i++ << " [m]";
//                    task->resetParameter(Task::Parameters(0., 1.5, 1.0, taskName.str()));
//                }
                
                tasks.emplace_back(new TaskPosition3D(graph, effector, goal1, nullptr));
                tasks.emplace_back(new TaskPosition3D(graph, effector, goal2, nullptr));
                tasks.emplace_back(new TaskPosition3D(graph, effector, goal3, nullptr));
                for (auto task : tasks) {
                    std::stringstream taskName;
                    taskName << "Position " << i++ << " [m]";
                    task->resetParameter(
                        Task::Parameters(-0.9, 0.9, 1.0, static_cast<std::string>("X") + taskName.str()));
                    task->addParameter(
                        Task::Parameters(-0.01, 0.01, 1.0, static_cast<std::string>("Y") + taskName.str()));
                    task->addParameter(Task::Parameters(-0., 1.4, 1.0, static_cast<std::string>("Z") + taskName.str()));
                }
            }
            else {
                tasks.emplace_back(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
                tasks.emplace_back(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
            }
            
            // Add the tasks
            for (auto t : tasks) { amIK->addTask(t); }
            
            // Incorporate collision costs into IK
            if (properties->getPropertyBool("collisionAvoidanceIK", true)) {
                REXEC(4) {
                    std::cout << "IK considers the provided collision model" << std::endl;
                }
                amIK->setupCollisionModel(collisionMdl);
            }
            
            return amIK;
        }
        else if (actionModelType == "ds_activation") {
            // Obtain the inner action model
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            
            // Check if the MPs are defined on position or task level
            if (properties->getPropertyBool("positionTasks", true)) {
                innerAM->addTask(new TaskPosition1D("X", graph, effector, nullptr, nullptr));
                innerAM->addTask(new TaskPosition1D("Z", graph, effector, nullptr, nullptr));
            }
            else {
                innerAM->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
                innerAM->addTask(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
            }
            
            // Obtain the task data
            auto& taskSpec = properties->getChildList("tasks");
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
            for (auto ts : taskSpec) {
                // All tasks cover the x and the z coordinate, thus no DSSlice is necessary
                tasks.emplace_back(DynamicalSystem::create(ts, innerAM->getDim()));
            }
            if (tasks.empty()) {
                throw std::invalid_argument("No tasks specified!");
            }
            
            // Incorporate collision costs into IK
            if (properties->getPropertyBool("collisionAvoidanceIK", true)) {
                REXEC(4) {
                    std::cout << "IK considers the provided collision model" << std::endl;
                }
                innerAM->setupCollisionModel(collisionMdl);
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
            
            // Create the action model
            return new AMDynamicalSystemActivation(innerAM.release(), taskRel, tcm);
        }
        else {
            std::ostringstream os;
            os << "Unsupported action model type: " << actionModelType;
            throw std::invalid_argument(os.str());
        }
    }
    
    virtual ObservationModel* createObservationModel()
    {
        auto fullState = new OMCombined();
        
        if (properties->getPropertyBool("observeVelocities", true)) {
            // Observe effector position and velocities
            auto omLin = new OMBodyStateLinear(graph, "Effector");  // in world coordinates
            omLin->setMinState(-1.56); // [m]
            omLin->setMaxState(1.56); // [m]
            omLin->setMaxVelocity(10.0); // [m/s]
            fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));  // mask out y axis
        }
        else {
            auto omLin = new OMBodyStateLinearPositions(graph, "Effector"); // in world coordinates
            omLin->setMinState(-1.56); // [m]
            omLin->setMaxState(1.56); // [m]
            fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));
        }
        
        std::string actionModelType = "joint_pos";
        properties->getProperty(actionModelType, "actionModelType");
        bool haveJointPos = actionModelType == "joint_pos";
        if (haveJointPos) {
            fullState->addPart(OMJointState::observeUnconstrainedJoints(graph));
        }
        else if (actionModelType == "ds_activation") {
            if (properties->getPropertyBool("observeDSGoalDistance", false)) {
                // Add goal distances
                auto castedAM = actionModel->unwrap<AMDynamicalSystemActivation>();
                if (castedAM) {
                    auto omGoalDist = new OMDynamicalSystemGoalDistance(castedAM);
                    fullState->addPart(omGoalDist);
                }
                else {
                    delete fullState;
                    std::ostringstream os;
                    os << "The action model needs to be of type AMDynamicalSystemActivation but is: " << castedAM;
                    throw std::invalid_argument(os.str());
                }
            }
            
            if (properties->getPropertyBool("observeDynamicalSystemDiscrepancy", false)) {
                // Add the discrepancies between commanded and executed the task space changes
                auto castedAM = dynamic_cast<AMDynamicalSystemActivation*>(actionModel);
                if (castedAM) {
                    auto omDescr = new OMDynamicalSystemDiscrepancy(castedAM);
                    fullState->addPart(omDescr);
                }
                else {
                    delete fullState;
                    std::ostringstream os;
                    os << "The action model needs to be of type AMDynamicalSystemActivation but is: " << castedAM;
                    throw std::invalid_argument(os.str());
                }
            }
        }
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true)) {
            RcsSensor* fts = RcsGraph_getSensorByName(graph, "EffectorLoadCell");
            if (fts) {
                auto omForceTorque = new OMForceTorque(graph, fts->name, 500);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, false, true, false, false, false}));
            }
        }
        
        // Add current collision cost
        if (properties->getPropertyBool("observeCollisionCost", false) & (collisionMdl != nullptr)) {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        // Add collision prediction
        if (properties->getPropertyBool("observePredictedCollisionCost", false) && collisionMdl != nullptr) {
            // Get horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add collision model
            auto omCollisionCost = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, 20);
            fullState->addPart(omCollisionCost);
        }
        
        // Add manipulability index
        auto ikModel = actionModel->unwrap<ActionModelIK>();
        if (properties->getPropertyBool("observeManipulabilityIndex", false) && ikModel) {
            bool ocm = properties->getPropertyBool("observeCurrentManipulability", true);
            fullState->addPart(new OMManipulabilityIndex(ikModel, ocm));
        }
        
        // Add the task space discrepancy observation model
        if (properties->getPropertyBool("observeTaskSpaceDiscrepancy", false)) {
            auto wamIK = actionModel->unwrap<ActionModelIK>();
            if (wamIK) {
                auto omTSDescr = new OMTaskSpaceDiscrepancy("Effector", graph, wamIK->getController()->getGraph());
                fullState->addPart(OMPartial::fromMask(omTSDescr, {true, false, true}));
            }
            else {
                delete fullState;
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Link1", new PPDMassProperties());
        manager->addParam("Link2", new PPDMassProperties());
        manager->addParam("Link3", new PPDMassProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSPlanar3Link(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* link3 = RcsGraph_getBodyByName(graph, "Link3");
        RCHECK(link3);
//        RcsBody* base = RcsGraph_getBodyByName(graph, "Base");
//        RCHECK(base);
//        return new ForceDisturber(link3, base);
        return new ForceDisturber(link3, link3);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set camera next to base
        RcsBody* base = RcsGraph_getBodyByName(graph, "Base");
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, base->A_BI->org);
        cameraCenter[1] -= 0.5;
        cameraCenter[2] += 0.3;
        
        // Set the camera position
        double cameraLocation[3];
        cameraLocation[0] = 0.;
        cameraLocation[1] = 4.;
        cameraLocation[2] = 2.5;
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        // Apply camera position
        viewer->setCameraHomePosition(osg::Vec3d(cameraLocation[0], cameraLocation[1], cameraLocation[2]),
                                      osg::Vec3d(cameraCenter[0], cameraCenter[1], cameraCenter[2]),
                                      osg::Vec3d(cameraUp[0], cameraUp[1], cameraUp[2]));
#endif
    }
    
    void
    getHUDText(
        std::vector<std::string>& linesOut,
        double currentTime,
        const MatNd* obs,
        const MatNd* currentAction,
        PhysicsBase* simulator,
        PhysicsParameterManager* physicsManager,
        ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != nullptr) {
            simName = simulator->getClassName();
        }
        
        linesOut.emplace_back(
            string_format("physics engine: %s                           sim time: %2.3f s", simName, currentTime));
        
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
        
        unsigned int sd = observationModel->getStateDim();
        
        auto omLin = observationModel->findOffsets<OMBodyStateLinear>();
        auto omLinPos = observationModel->findOffsets<OMBodyStateLinearPositions>();
        if (omLin) {
            linesOut.emplace_back(
                string_format("end-eff pos:   [% 1.3f,% 1.3f] m  end-eff vel:   [% 1.2f,% 1.2f] m/s",
                              obs->ele[omLin.pos], obs->ele[omLin.pos + 1],
                              obs->ele[sd + omLin.vel], obs->ele[sd + omLin.vel + 1]));
        }
        else if (omLinPos) {
            linesOut.emplace_back(
                string_format("end-eff pos:   [% 1.3f,% 1.3f] m",
                              obs->ele[omLinPos.pos], obs->ele[omLinPos.pos + 1]));
        }
        
        auto omGD = observationModel->findOffsets<OMDynamicalSystemGoalDistance>();
        if (omGD) {
            linesOut.emplace_back(
                string_format("goal dist pos: [% 1.3f,% 1.3f,% 1.3f] m",
                              obs->ele[omGD.pos + 0], obs->ele[omGD.pos + 1], obs->ele[omGD.pos + 2]));
        }
        
        auto omFT = observationModel->findOffsets<OMForceTorque>();
        if (omFT) {
            linesOut.emplace_back(
                string_format("forces:        [% 3.1f,% 3.1f] N", obs->ele[omFT.pos + 0], obs->ele[omFT.pos + 1]));
        }
        
        const double* distForce = forceDisturber->getLastForce();
        linesOut.emplace_back(
            string_format("disturbances:  [% 3.1f,% 3.1f,% 3.1f] N", distForce[0], distForce[1], distForce[2]));
        
        
        linesOut.emplace_back(
            string_format("actions:       [% 1.3f,% 1.3f,% 1.3f]",
                          currentAction->ele[0], currentAction->ele[1], currentAction->ele[2]));
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* link1_bpi = physicsManager->getBodyInfo("Link1");
            BodyParamInfo* link2_bpi = physicsManager->getBodyInfo("Link2");
            BodyParamInfo* link3_bpi = physicsManager->getBodyInfo("Link3");
            
            linesOut.emplace_back(
                string_format("link masses:   [% 1.3f,% 1.3f,% 1.3f] kg", link1_bpi->body->m, link2_bpi->body->m,
                              link3_bpi->body->m));
        }
        
        auto omManip = observationModel->findOffsets<OMManipulabilityIndex>();
        if (omManip) {
            linesOut.emplace_back(string_format("manipulability: % 5.3f", obs->ele[omManip.pos + 0]));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECPlanar3Link> RegPlanar3Link("Planar3Link");

}

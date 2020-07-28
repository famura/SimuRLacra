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
#include "action/AMIKControllerActivation.h"
#include "action/AMDynamicalSystemActivation.h"
#include "initState/ISSBallInTube.h"
#include "observation/OMCombined.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMPartial.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "physics/PPDSphereRadius.h"
#include "util/string_format.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_Vec3d.h>
#include <TaskPosition3D.h>
#include <TaskPosition1D.h>
#include <TaskVelocity1D.h>
#include <TaskPositionForce1D.h>
#include <TaskOmega1D.h>
#include <TaskEuler3D.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <memory>
#include <iomanip>

namespace Rcs
{
class ECBallInTube : public ExperimentConfig
{
protected:
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* leftEffector = RcsGraph_getBodyByName(graph, "Effector_L");
        RCHECK(leftEffector);
        RcsBody* rightEffector = RcsGraph_getBodyByName(graph, "Effector_R");
        RCHECK(rightEffector);
    
        // Get reference frames for the position and orientation tasks
        std::string refFrameType = "world";
        properties->getProperty(refFrameType, "refFrame");
        RcsBody* refBody = nullptr;
        RcsBody* refFrame = nullptr;
        if (refFrameType == "world") {
            // Keep nullptr
        }
        else if (refFrameType == "table") {
            RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
            RCHECK(table);
            refBody = table;
            refFrame = table;
        }
        else if (refFrameType == "slider") {
            RcsBody* slider = RcsGraph_getBodyByName(graph, "Slider");
            RCHECK(slider);
            refBody = slider;
            refFrame = slider;
        }
        else {
            std::ostringstream os;
            os << "Unsupported reference frame type: " << refFrame;
            throw std::invalid_argument(os.str());
        }
    
        // Get the method how to combine the movement primitives / tasks given their activation
        std::string taskCombinationMethod = "mean";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(taskCombinationMethod);
    
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
    
        if (actionModelType == "ik") {
            // Create the action model
            auto amIK = new AMIKGeneric(graph);
            std::vector<Task*> tasks;
        
            if (properties->getPropertyBool("positionTasks", true)) {
                throw std::invalid_argument("Position tasks are not implemented for AMIKGeneric in this environment.");
            }
            else {
                // Left
                tasks.emplace_back(new TaskVelocity1D("Xd", graph, leftEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt, dt, 1.0, "Xd Left [m/s]"));
                tasks.emplace_back(new TaskVelocity1D("Yd", graph, leftEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt, dt, 1.0, "Yd Left [m/s]"));
                tasks.emplace_back(new TaskVelocity1D("Zd", graph, leftEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt, dt, 1.0, "Zd Left [m/s]"));
                tasks.emplace_back(new TaskOmega1D("Ad", graph, leftEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt*M_PI_2, dt*M_PI_2, 1.0, "Ad Left [deg/s]"));
                tasks.emplace_back(new TaskOmega1D("Bd", graph, leftEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt*M_PI_2, dt*M_PI_2, 1.0, "Bd Left [deg/s]"));
                tasks.emplace_back(new TaskOmega1D("Cd", graph, leftEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt*M_PI_2, dt*M_PI_2, 1.0, "Cd Left [deg/s]"));
            
                // Right
                tasks.emplace_back(new TaskVelocity1D("Xd", graph, rightEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt, dt, 1.0, "Xd Right [m/s]"));
                tasks.emplace_back(new TaskVelocity1D("Yd", graph, rightEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt, dt, 1.0, "Yd Right [m/s]"));
                tasks.emplace_back(new TaskVelocity1D("Zd", graph, rightEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt, dt, 1.0, "Zd Right [m/s]"));
                tasks.emplace_back(new TaskOmega1D("Ad", graph, rightEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt*M_PI_2, dt*M_PI_2, 1.0, "Ad Right [deg/s]"));
                tasks.emplace_back(new TaskOmega1D("Bd", graph, rightEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt*M_PI_2, dt*M_PI_2, 1.0, "Bd Right [deg/s]"));
                tasks.emplace_back(new TaskOmega1D("Cd", graph, rightEffector, refBody, refFrame));
                tasks.back()->resetParameter(Task::Parameters(-dt*M_PI_2, dt*M_PI_2, 1.0, "Cd Right [deg/s]"));
            
                // Add the tasks
                for (auto t : tasks) { amIK->addTask(t); }
            
                return amIK;
            }
        }
    
        else if (actionModelType == "ik_activation") {
            // Create the action model
            auto amIK = new AMIKControllerActivation(graph, tcm);
            std::vector<Task*> tasks;
        
            if (properties->getPropertyBool("positionTasks", true)) {
                RcsBody* ball = RcsGraph_getBodyByName(graph, "Ball");
                RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
                RcsBody* slider = RcsGraph_getBodyByName(graph, "Slider");
                RCHECK(ball);
                RCHECK(table);
                RCHECK(slider);
                std::string taskName;
            
                // Left
                auto tl0 = new TaskPosition3D(graph, leftEffector, nullptr, nullptr);
                taskName = " Position Home [m]";
                tl0->resetParameter(Task::Parameters(0., 2., 1.0, "X" + taskName));
                tl0->addParameter(Task::Parameters(-1., 1., 1.0, "Y" + taskName));
                tl0->addParameter(Task::Parameters(0., 1.7, 1.0, "Z" + taskName));
                tasks.emplace_back(tl0);
                auto tl1 = new TaskPosition3D(graph, leftEffector, ball, nullptr);
                taskName = " Position rel Ball [m]";
                tl1->resetParameter(Task::Parameters(-2., 2., 1.0, "X" + taskName));
                tl1->addParameter(Task::Parameters(-1., 1., 1.0, "Y" + taskName));
                tl1->addParameter(Task::Parameters(
                    -ball->shape[0]->extents[0], 1., 1.0, "Z" + taskName));
                tasks.emplace_back(tl1);
//                RCHECK(RcsGraph_getSensorByName(graph, "lbr_joint_7_torque_L"));
//                auto tl2 = new TaskPositionForce1D("ForceX", graph, leftEffector, ball, nullptr, "lbr_joint_7_torque_L", false);
//                tasks.emplace_back(tl2);
                auto tl3 = new TaskEuler3D(graph, leftEffector, table, nullptr);
                tasks.emplace_back(tl3);
                // Right
                auto tr0 = new TaskPosition3D(graph, rightEffector, nullptr, nullptr);
                taskName = " Position Home [m]";
                tr0->resetParameter(Task::Parameters(0., 2., 1.0, "X" + taskName));
                tr0->addParameter(Task::Parameters(-1., 1., 1.0, "Y" + taskName));
                tr0->addParameter(Task::Parameters(0., 1.7, 1.0, "Z" + taskName));
                tasks.emplace_back(tr0);
                auto tr1 = new TaskPosition3D(graph, rightEffector, slider, nullptr);
                taskName = " Position rel Slider [m]";
                tr1->resetParameter(Task::Parameters(-2., 2., 1.0, "X" + taskName));
                tr1->addParameter(Task::Parameters(-1., 1., 1.0, "Y" + taskName));
                tr1->addParameter(Task::Parameters(-0.5, 1., 1.0, "Z" + taskName));
                tasks.emplace_back(tr1);
                auto tr2 = new TaskPosition1D("Y", graph, rightEffector, table, nullptr);
                tr2->resetParameter(Task::Parameters(-1., 1., 1.0, "Y Position [m]"));
                tasks.emplace_back(tr2);
                auto tr3 = new TaskEuler3D(graph, rightEffector, slider, nullptr);
                tasks.emplace_back(tr3);
            
                // Add the tasks
                for (auto t : tasks) { amIK->addTask(t); }
            
                // Set the tasks' desired states
                std::vector<PropertySource*> taskSpec = properties->getChildList("taskSpecIK");
                amIK->setXdesFromTaskSpec(taskSpec, tasks);
            
                // Incorporate collision costs into IK
                if (properties->getPropertyBool("collisionAvoidanceIK", true)) {
                    REXEC(4) {
                        std::cout << "IK considers the provided collision model" << std::endl;
                    }
                    amIK->setupCollisionModel(collisionMdl);
                }
            }
            else {
                throw std::invalid_argument("Velocity tasks are not supported for AMIKControllerActivation.");
            }
        
            return amIK;
        }
    
        else if (actionModelType == "ds_activation") {
            // Initialize action model and tasks
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
        
            // Control effector positions and orientation
            if (properties->getPropertyBool("positionTasks", false)) {
                RcsBody* slider = RcsGraph_getBodyByName(graph, "Slider");
                RCHECK(slider);
                // Left
                innerAM->addTask(new TaskPosition3D(graph, leftEffector, slider, slider));
                innerAM->addTask(new TaskEuler3D(graph, leftEffector, slider, slider));
                // Right
                innerAM->addTask(new TaskPosition3D(graph, rightEffector, slider, slider));
                innerAM->addTask(new TaskEuler3D(graph, rightEffector, slider, slider));
    
                // Obtain task data (depends on the order of the MPs coming from Pyrado)
                // Left
                unsigned int i = 0;
                std::vector<unsigned int> taskDimsLeft{
                    3, 3, 3, 3
                };
                std::vector<unsigned int> offsetsLeft{
                    0, 0, 3, 3
                };
                auto& tsLeft = properties->getChildList("tasksLeft");
                for (auto tsk : tsLeft) {
                    DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                    tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                    i++;
                }
                // Right
                std::vector<unsigned int> taskDimsRight{
                    3, 3, 3, 3
                };
                unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
                std::vector<unsigned int> offsetsRight{
                    oL, oL, oL + 3, oL + 3
                };
                i = 0;
                auto& tsRight = properties->getChildList("tasksRight");
                for (auto tsk : tsRight) {
                    DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsRight[i]);
                    tasks.emplace_back(new DSSlice(ds, offsetsRight[i], taskDimsRight[i]));
                    i++;
                }
            }
    
                // Control effector velocity and orientation
            else {
                // Left
                innerAM->addTask(new TaskVelocity1D("Xd", graph, leftEffector, refBody, refFrame));
                innerAM->addTask(new TaskVelocity1D("Yd", graph, leftEffector, refBody, refFrame));
                innerAM->addTask(new TaskVelocity1D("Zd", graph, leftEffector, refBody, refFrame));
                innerAM->addTask(new TaskOmega1D("Ad", graph, leftEffector, refBody, refFrame));
                innerAM->addTask(new TaskOmega1D("Bd", graph, leftEffector, refBody, refFrame));
                innerAM->addTask(new TaskOmega1D("Cd", graph, leftEffector, refBody, refFrame));
                // Right
                innerAM->addTask(new TaskVelocity1D("Xd", graph, rightEffector, refBody, refFrame));
                innerAM->addTask(new TaskVelocity1D("Yd", graph, rightEffector, refBody, refFrame));
                innerAM->addTask(new TaskVelocity1D("Zd", graph, rightEffector, refBody, refFrame));
                innerAM->addTask(new TaskOmega1D("Ad", graph, rightEffector, refBody, refFrame));
                innerAM->addTask(new TaskOmega1D("Bd", graph, rightEffector, refBody, refFrame));
                innerAM->addTask(new TaskOmega1D("Cd", graph, rightEffector, refBody, refFrame));
    
                // Obtain task data (depends on the order of the MPs coming from Pyrado)
                // Left
                unsigned int i = 0;
                std::vector<unsigned int> taskDimsLeft{
                    1, 1, 1, 1, 1, 1
                };
                std::vector<unsigned int> offsetsLeft{
                    0, 1, 2, 3, 4, 5
                };
                auto& tsLeft = properties->getChildList("tasksLeft");
                for (auto tsk : tsLeft) {
                    DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                    tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                    i++;
                }
                // Right
                std::vector<unsigned int> taskDimsRight{
                    1, 1, 1, 1, 1, 1
                };
                unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
                std::vector<unsigned int> offsetsRight{
                    oL, oL + 1, oL + 2, oL + 3, oL + 4, oL + 5
                };
                i = 0;
                auto& tsRight = properties->getChildList("tasksRight");
                for (auto tsk : tsRight) {
                    DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsRight[i]);
                    tasks.emplace_back(new DSSlice(ds, offsetsRight[i], taskDimsRight[i]));
                    i++;
                }
            }
        
            if (tasks.empty()) {
                throw std::invalid_argument("No tasks specified!");
            }
        
            // Incorporate collision costs into IK
            if (properties->getPropertyBool("collisionAvoidanceIK", true)) {
                std::cout << "IK considers the provided collision model" << std::endl;
                innerAM->setupCollisionModel(collisionMdl);
            }
        
            // Setup task-based action model
            std::vector<DynamicalSystem*> taskRel;
            for (auto& task : tasks) {
                taskRel.push_back(task.release());
            }
        
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
        // Observe effector positions (and velocities)
        std::unique_ptr<OMCombined> fullState(new OMCombined());
        
        if (properties->getPropertyBool("observeVelocities", true)) {
            auto omLeftLin = new OMBodyStateLinear(graph, "Effector_L"); // in world coordinates
            omLeftLin->setMinState({0.2, -1., 0.74});  // [m]
            omLeftLin->setMaxState({1.8, 1., 1.5});  // [m]
            omLeftLin->setMaxVelocity(3.); // [m/s]
            fullState->addPart(omLeftLin);
            
            auto omRightLin = new OMBodyStateLinear(graph, "Effector_R"); // in world coordinates
            omRightLin->setMinState({0.2, -1., 0.74});  // [m]
            omRightLin->setMaxState({1.8, 1., 1.5});  // [m]
            omRightLin->setMaxVelocity(3.); // [m/s]
            fullState->addPart(omRightLin);
        }
        else {
            auto omLeftLin = new OMBodyStateLinearPositions(graph, "Effector_L"); // in world coordinates
            omLeftLin->setMinState({0.2, -1., 0.74});  // [m]
            omLeftLin->setMaxState({1.8, 1., 1.5});  // [m]
            fullState->addPart(omLeftLin);
            
            auto omRightLin = new OMBodyStateLinearPositions(graph, "Effector_R"); // in world coordinates
            omRightLin->setMinState({0.2, -1., 0.74});  // [m]
            omRightLin->setMaxState({1.8, 1., 1.5});  // [m]
            fullState->addPart(omRightLin);
        }
        
        // Observe ball positions (and velocities)
        if (properties->getPropertyBool("observeVelocities", true)) {
            auto omBallLin = new OMBodyStateLinear(graph, "Ball", nullptr, nullptr);  // in world coordinates
            omBallLin->setMinState({0.6, -1., 0.74});  // [m]
            omBallLin->setMaxState({1.6, 1., 0.88});  // [m]
            omBallLin->setMaxVelocity(5.); // [m/s]
            fullState->addPart(OMPartial::fromMask(omBallLin, {true, true, false})); // only x, y component
        }
        else {
            auto omBallLin = new OMBodyStateLinearPositions(graph, "Ball", nullptr, nullptr);  // in world coordinates
            omBallLin->setMinState({0.6, -1., 0.7});  // [m]
            omBallLin->setMaxState({1.6, 1., 0.9});  // [m]
            fullState->addPart(OMPartial::fromMask(omBallLin, {true, true, false})); // only x, y component
        }
        
        // Observe slider position (and velocities)
        if (properties->getPropertyBool("observeVelocities", true)) {
            auto omSliderLin = new OMBodyStateLinear(graph, "Slider", nullptr, nullptr);  // in world coordinates
            omSliderLin->setMinState({0.9, -1., 0.7});  // [m]
            omSliderLin->setMaxState({1.3, 0., 0.9});  // [m]
            omSliderLin->setMaxVelocity(5.); // [m/s]
            fullState->addPart(OMPartial::fromMask(omSliderLin, {false, true, false})); // only y component
        }
        else {
            auto omSliderLin = new OMBodyStateLinearPositions(graph, "Slider", nullptr,
                                                              nullptr);  // in world coordinates
            omSliderLin->setMinState({0.9, -1., 0.7});  // [m]
            omSliderLin->setMaxState({1.3, 0., 0.9});  // [m]
            fullState->addPart(OMPartial::fromMask(omSliderLin, {false, true, false})); // only y component
        }
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true)) {
            RcsSensor* ftsL = RcsGraph_getSensorByName(graph, "WristLoadCellLBR_L");
            if (ftsL) {
                auto omForceTorque = new OMForceTorque(graph, ftsL->name, 1200);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, true, true, false, false, false}));
            }
            RcsSensor* ftsR = RcsGraph_getSensorByName(graph, "WristLoadCellLBR_R");
            if (ftsR) {
                auto omForceTorque = new OMForceTorque(graph, ftsR->name, 1200);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, true, true, false, false, false}));
            }
        }
        
        // Add current collision cost
        if (properties->getPropertyBool("observeCollisionCost", true) & (collisionMdl != nullptr)) {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        // Add predicted collision cost
        if (properties->getPropertyBool("observePredictedCollisionCost", false) & (collisionMdl != nullptr)) {
            // Get the horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add the collision cost observation model
            auto omCollPred = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, horizon);
            fullState->addPart(omCollPred);
        }
        
        // Add manipulability index
        auto ikModel = actionModel->unwrap<ActionModelIK>();
        if (properties->getPropertyBool("observeManipulabilityIndex", false) && ikModel) {
            bool ocm = properties->getPropertyBool("observeCurrentManipulability", true);
            fullState->addPart(new OMManipulabilityIndex(ikModel, ocm));
        }
        
        // Add the task space discrepancy observation model
        if (properties->getPropertyBool("observeTaskSpaceDiscrepancy", true)) {
            auto wamIK = actionModel->unwrap<ActionModelIK>();
            if (wamIK) {
                auto omTSDescrL = new OMTaskSpaceDiscrepancy("Effector_L", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescrL);
                auto omTSDescrR = new OMTaskSpaceDiscrepancy("Effector_R", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescrR);
            }
            else {
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        if (actionModelType == "ds_activation") {
            // Add goal distances
            if (properties->getPropertyBool("observeDynamicalSystemGoalDistance", false)) {
                auto amAct = actionModel->unwrap<AMDynamicalSystemActivation>();
                RCHECK(amAct);
                fullState->addPart(new OMDynamicalSystemGoalDistance(amAct));
            }
    
            // Add the dynamical system discrepancy observation model
            if (properties->getPropertyBool("observeDynamicalSystemDiscrepancy", false) & (collisionMdl != nullptr)) {
                auto castedAM = dynamic_cast<AMDynamicalSystemActivation*>(actionModel);
                if (castedAM) {
                    auto omDSDescr = new OMDynamicalSystemDiscrepancy(castedAM);
                    fullState->addPart(omDSDescr);
                }
                else {
                    throw std::invalid_argument("The action model needs to be of type AMDynamicalSystemActivation!");
                }
            }
        }
        
        return fullState.release();
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Ball", new PPDSphereRadius("Table"));
        manager->addParam("Ball", new PPDMassProperties());
        manager->addParam("Ball", new PPDMaterialProperties());
        manager->addParam("Slider", new PPDMassProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSBallInTube(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* box = RcsGraph_getBodyByName(graph, "Ball");
        RCHECK(box);
        return new ForceDisturber(box, box);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set camera above plate
        RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
        RCHECK(table);
        std::string cameraView = "egoView";
        properties->getProperty(cameraView, "egoView");
        
        // The camera center is 10cm above the the plate's center
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, table->A_BI->org);
        cameraCenter[0] -= 0.2;
        
        // The camera location - not specified yet
        double cameraLocation[3];
        Vec3d_setZero(cameraLocation);
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        if (cameraView == "egoView") {
            RcsBody* railBot = RcsGraph_getBodyByName(graph, "RailBot");
            RCHECK(railBot);
            
            // Rotate to world frame
            Vec3d_transformSelf(cameraLocation, table->A_BI);
            
            // Move the camera approx where the Kinect would be
            cameraLocation[0] = railBot->A_BI->org[0] - 0.5;
            cameraLocation[1] = railBot->A_BI->org[1];
            cameraLocation[2] = railBot->A_BI->org[2] + 1.5;
        }
        else {
            RMSG("Unsupported camera view: %s", cameraView.c_str());
            return;
        }
        
        // Apply the camera position
        viewer->setCameraHomePosition(osg::Vec3d(cameraLocation[0], cameraLocation[1], cameraLocation[2]),
                                      osg::Vec3d(cameraCenter[0], cameraCenter[1], cameraCenter[2]),
                                      osg::Vec3d(cameraUp[0], cameraUp[1], cameraUp[2]));
#endif
    }
    
    void
    getHUDText(
        std::vector<std::string>& linesOut,
        double currentTime, const MatNd* obs,
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
            string_format("physics engine: %s                     sim time:        %2.3f s", simName, currentTime));
    
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
    
        auto omLeftLin = observationModel->findOffsets<OMBodyStateLinear>(); // there are two, we find the first
        if (omLeftLin) {
            linesOut.emplace_back(
                string_format("left hand pg:  [% 1.3f,% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f,% 1.3f] m/s",
                              obs->ele[omLeftLin.pos], obs->ele[omLeftLin.pos + 1], obs->ele[omLeftLin.pos + 2],
                              obs->ele[sd + omLeftLin.vel], obs->ele[sd + omLeftLin.vel + 1],
                              obs->ele[sd + omLeftLin.vel + 2]));
            linesOut.emplace_back(
                string_format("right hand pg: [% 1.3f,% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f,% 1.3f] m/s",
                              obs->ele[omLeftLin.pos + 3], obs->ele[omLeftLin.pos + 4], obs->ele[omLeftLin.pos + 5],
                              obs->ele[sd + omLeftLin.vel + 3], obs->ele[sd + omLeftLin.vel + 4],
                              obs->ele[sd + omLeftLin.vel + 5]));
        }

        else if (omLeftLin) {
            linesOut.emplace_back(
                string_format("box absolute:  [% 1.3f,% 1.3f,% 1.3f] m",
                              obs->ele[omLeftLin.pos + 6], obs->ele[omLeftLin.pos + 7], obs->ele[omLeftLin.pos + 8]));
        }
    
        auto omFTS = observationModel->findOffsets<OMForceTorque>();
        if (omFTS) {
            linesOut.emplace_back(
                string_format("forces left:   [% 3.1f, % 3.1f, % 3.1f] N     right: [% 3.1f, % 3.1f, % 3.1f] N",
                              obs->ele[omFTS.pos], obs->ele[omFTS.pos + 1], obs->ele[omFTS.pos + 2],
                              obs->ele[omFTS.pos + 3], obs->ele[omFTS.pos + 4], obs->ele[omFTS.pos + 5]));
        }
    
        auto omColl = observationModel->findOffsets<OMCollisionCost>();
        auto omCollPred = observationModel->findOffsets<OMCollisionCostPrediction>();
        if (omColl && omCollPred) {
            linesOut.emplace_back(
                string_format("coll cost:       %3.2f                    pred coll cost: %3.2f",
                              obs->ele[omColl.pos], obs->ele[omCollPred.pos]));
    
        }
        else if (omColl) {
            linesOut.emplace_back(string_format("coll cost:       %3.2f", obs->ele[omColl.pos]));
        }
        else if (omCollPred) {
            linesOut.emplace_back(string_format("pred coll cost:   %3.2f", obs->ele[omCollPred.pos]));
        }
    
        auto omMI = observationModel->findOffsets<OMManipulabilityIndex>();
        if (omMI) {
            linesOut.emplace_back(string_format("manip idx:       %1.3f", obs->ele[omMI.pos]));
        }
    
        auto omGD = observationModel->findOffsets<OMDynamicalSystemGoalDistance>();
        if (omGD) {
            if (properties->getPropertyBool("positionTasks", false)) // TODO
            {
                linesOut.emplace_back(
                    string_format("goal distance: [% 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f,\n"
                                  "               % 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f]",
                                  obs->ele[omGD.pos], obs->ele[omGD.pos + 1], obs->ele[omGD.pos + 2],
                                  obs->ele[omGD.pos + 3], obs->ele[omGD.pos + 4],
                                  obs->ele[omGD.pos + 5], obs->ele[omGD.pos + 6], obs->ele[omGD.pos + 7],
                                  obs->ele[omGD.pos + 8], obs->ele[omGD.pos + 9], obs->ele[omGD.pos + 10]));
            }
        }
    
        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD) {
            linesOut.emplace_back(
                string_format("ts delta:      [% 1.3f,% 1.3f,% 1.3f] m",
                              obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1], obs->ele[omTSD.pos + 2]));
        }
    
        std::stringstream ss;
        ss << "actions:       [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++) {
            ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6) {
                ss << "\n                ";
            }
        }
        ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
    
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* ball_bpi = physicsManager->getBodyInfo("Ball");
            double* com = ball_bpi->body->Inertia->org;
            double ball_radius = ball_bpi->body->shape[0]->extents[0];
            double slip = 0;
            ball_bpi->material.getDouble("slip", slip);
        
            linesOut.emplace_back(string_format(
                "ball mass:      %2.2f kg           ball radius:             %2.3f cm",
                ball_bpi->body->m, ball_radius*100));
        
            linesOut.emplace_back(string_format(
                "ball friction:  %1.2f    ball rolling friction:             %1.3f",
                ball_bpi->material.getFrictionCoefficient(),
                ball_bpi->material.getRollingFrictionCoefficient()/ball_radius));
        
            linesOut.emplace_back(string_format(
                "ball slip:      %3.1f rad/(Ns)       CoM offset:[% 2.1f, % 2.1f, % 2.1f] mm",
                slip, com[0]*1000, com[1]*1000, com[2]*1000));
        }
    }
    
};

// Register
static ExperimentConfigRegistration<ECBallInTube> RegBallInTube("BallInTube");

}

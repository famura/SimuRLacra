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
#include "action/AMIKControllerActivation.h"
#include "initState/ISSBoxFlipping.h"
#include "observation/OMCombined.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMPartial.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDBoxExtents.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_Vec3d.h>
#include <TaskPosition1D.h>
#include <TaskVelocity1D.h>
#include <TaskDistance.h>
#include <TaskDistance1D.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <memory>
#include <iomanip>

namespace Rcs
{
class ECBoxFlipping : public ExperimentConfig
{
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* leftCP = RcsGraph_getBodyByName(graph, "ContactPoint_L");
        RCHECK(leftCP);
        RcsBody* rightCP = RcsGraph_getBodyByName(graph, "ContactPoint_R");
        RCHECK(rightCP);
        RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
        RCHECK(table);
        RcsBody* box = RcsGraph_getBodyByName(graph, "Box");
        RCHECK(box);
        
        // Get reference frames for the position and orientation tasks
        std::string refFrameType = "world";
        properties->getProperty(refFrameType, "refFrame");
        RcsBody* refBody = nullptr;
        RcsBody* refFrame = nullptr;
        if (refFrameType == "world") {
            // Keep nullptr
        }
        else if (refFrameType == "box") {
            RcsBody* box = RcsGraph_getBodyByName(graph, "Box");
            RCHECK(box);
            RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
            RCHECK(table);
            refBody = box;
            refFrame = table; // do not use a moving frame
        }
        else if (refFrameType == "table") {
            RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
            RCHECK(table);
            refBody = table;
            refFrame = table;
        }
        else {
            std::ostringstream os;
            os << "Unsupported reference frame type: " << refFrame;
            throw std::invalid_argument(os.str());
        }
        
        // Get the type of action model
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Get the method how to combine the movement primitives / tasks given their activation (common for both)
        std::string taskCombinationMethod = "unspecified";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(taskCombinationMethod);
        
        if (actionModelType == "ik_activation") {
            // Create the action model
            auto amIK = new AMIKControllerActivation(graph, tcm);
            std::vector<Task*> tasks;
            
            // Check if the tasks are defined on position or velocity level. Adapt their parameters if desired.
            if (properties->getPropertyBool("positionTasks", true)) {
                // Left
                tasks.emplace_back(new TaskPosition1D("Y", graph, leftCP, refBody, refFrame));
                tasks.emplace_back(new TaskPosition1D("Z", graph, leftCP, refBody, refFrame));
//                tasks.emplace_back(new TaskDistance1D(graph, leftCP, box, 1));  // Y
                // Right
                tasks.emplace_back(new TaskPosition1D("Y", graph, rightCP, refBody, refFrame));
                tasks.emplace_back(new TaskPosition1D("Z", graph, rightCP, refBody, refFrame));
//                tasks.emplace_back(new TaskDistance1D(graph, rightCP, box, 1));  // Y
            }
            else {
                // Left
                tasks.emplace_back(new TaskVelocity1D("Yd", graph, leftCP, refBody, refFrame));
                tasks.emplace_back(new TaskVelocity1D("Zd", graph, leftCP, refBody, refFrame));
                // Right
                tasks.emplace_back(new TaskVelocity1D("Yd", graph, rightCP, refBody, refFrame));
                tasks.emplace_back(new TaskVelocity1D("Zd", graph, rightCP, refBody, refFrame));
            }
            
            // Add the tasks
            for (auto t : tasks) { amIK->addTask(t); }
            
            // Set the tasks' desired states
            std::vector<PropertySource*> taskSpec = properties->getChildList("taskSpecIK");
            amIK->setXdesFromTaskSpec(taskSpec);
            
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
            // Initialize action model and tasks
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
            
            // Control effector positions and orientation
            if (properties->getPropertyBool("positionTasks", false)) {
                // Left
                innerAM->addTask(new TaskPosition1D("Y", graph, leftCP, refBody, refFrame));
                innerAM->addTask(new TaskPosition1D("Z", graph, leftCP, refBody, refFrame));
                // Right
                //            innerAM->addTask(new TaskPosition1D("Y", graph, rightCP, refBody, refFrame));
                innerAM->addTask(new TaskPosition1D("Z", graph, rightCP, refBody, refFrame));
                innerAM->addTask(new TaskDistance(graph, rightCP, box));
                
                // Obtain task data (depends on the order of the MPs coming from Pyrado)
                // Left
                unsigned int i = 0;
                std::vector<unsigned int> taskDimsLeft{1, 1, 1, 1};
                std::vector<unsigned int> offsetsLeft{0, 0, 1, 1};
                auto& tsLeft = properties->getChildList("tasksLeft");
                for (auto tsk : tsLeft) {
                    DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                    tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                    i++;
                }
                // Right
                i = 0;
                //            std::vector<unsigned int> taskDimsRight{1, 1, 1, 1, 1};
                std::vector<unsigned int> taskDimsRight{1, 1};
                unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
                //            std::vector<unsigned int> offsetsRight{oL, oL, oL + 1, oL + 1, oL + 2};
                std::vector<unsigned int> offsetsRight{oL, oL + 1};
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
                innerAM->addTask(new TaskVelocity1D("Yd", graph, leftCP, refBody, refFrame));
                innerAM->addTask(new TaskVelocity1D("Zd", graph, leftCP, refBody, refFrame));
                // Right
                innerAM->addTask(new TaskVelocity1D("Yd", graph, rightCP, refBody, refFrame));
                innerAM->addTask(new TaskVelocity1D("Zd", graph, rightCP, refBody, refFrame));
                
                // Obtain task data (depends on the order of the MPs coming from Pyrado)
                // Left
                unsigned int i = 0;
                std::vector<unsigned int> taskDimsLeft{1, 1, 1, 1};
                std::vector<unsigned int> offsetsLeft{0, 0, 1, 1};
                auto& tsLeft = properties->getChildList("tasksLeft");
                for (auto tsk : tsLeft) {
                    DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                    tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                    i++;
                }
                // Right
                i = 0;
                std::vector<unsigned int> taskDimsRight{1, 1, 1, 1};
                unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
                std::vector<unsigned int> offsetsRight{oL, oL, oL + 1, oL + 1};
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
        
        if (properties->getPropertyBool("observeVelocities", false)) {
            // Left
            auto omLeftLin = new OMBodyStateLinear(graph, "ContactPoint_L"); // in world coordinates
            omLeftLin->setMinState({0., -1., 0.75});  // [m]
            omLeftLin->setMaxState({1.6, 1., 1.5});  // [m]
            omLeftLin->setMaxVelocity(2.); // [m/s]
            fullState->addPart(OMPartial::fromMask(omLeftLin, {false, true, true}));
            // Right
            auto omRightLin = new OMBodyStateLinear(graph, "ContactPoint_R"); // in world coordinates
            omRightLin->setMinState({0., -1., 0.75});  // [m]
            omRightLin->setMaxState({1.6, 1., 1.5});  // [m]
            omRightLin->setMaxVelocity(2.); // [m/s]
            fullState->addPart(OMPartial::fromMask(omRightLin, {false, true, true}));
        }
        else {
            // Left
            auto omLeftLin = new OMBodyStateLinearPositions(graph, "ContactPoint_L"); // in world coordinates
            omLeftLin->setMinState({0., -1., 0.75});  // [m]
            omLeftLin->setMaxState({1.6, 1., 1.5});  // [m]
            fullState->addPart(OMPartial::fromMask(omLeftLin, {false, true, true}));
            // Right
            auto omRightLin = new OMBodyStateLinearPositions(graph, "ContactPoint_R"); // in world coordinates
            omRightLin->setMinState({0., -1., 0.75});  // [m]
            omRightLin->setMaxState({1.6, 1., 1.5});  // [m]
            fullState->addPart(OMPartial::fromMask(omRightLin, {false, true, true}));
        }
        
        // Observe box positions (and velocities)
        if (properties->getPropertyBool("observeVelocities", false)) {
            auto omBoxLin = new OMBodyStateLinear(graph, "Box"); // in world coordinates
            omBoxLin->setMinState({0.9, -0.8, 0.66});  // [m]
            omBoxLin->setMaxState({2.1, 0.8, 1.26});  // [m]
            omBoxLin->setMaxVelocity(2.); // [m/s]
            fullState->addPart(OMPartial::fromMask(omBoxLin, {false, true, true}));
        }
        else {
            auto omBoxLin = new OMBodyStateLinearPositions(graph, "Box"); // in world coordinates
            omBoxLin->setMinState({0.9, -0.8, 0.66});  // [m]
            omBoxLin->setMaxState({2.1, 0.8, 1.26});  // [m]
            fullState->addPart(OMPartial::fromMask(omBoxLin, {false, true, true}));
        }
        
        // Observe box orientation (and velocities)
        if (properties->getPropertyBool("observeVelocities", false)) {
            auto omBoxAng = new OMBodyStateAngular(graph, "Box"); // in world coordinates
            omBoxAng->setMaxVelocity(RCS_DEG2RAD(720)); // [rad/s]
            fullState->addPart(OMPartial::fromMask(omBoxAng, {true, false, false}));
        }
        else {
            auto omBoxAng = new OMBodyStateAngularPositions(graph, "Box"); // in world coordinates
            fullState->addPart(OMPartial::fromMask(omBoxAng, {true, false, false}));
        }
        
        // Add goal distances
        if (properties->getPropertyBool("observeDynamicalSystemGoalDistance", false)) {
            auto amAct = actionModel->unwrap<AMDynamicalSystemActivation>();
            RCHECK(amAct);
            fullState->addPart(new OMDynamicalSystemGoalDistance(amAct));
        }
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true)) {
            RcsSensor* ftsL = RcsGraph_getSensorByName(graph, "WristLoadCell_L");
            if (ftsL) {
                auto omForceTorque = new OMForceTorque(graph, ftsL->name, 10000);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {false, true, true, false, false, false}));
            }
            RcsSensor* ftsR = RcsGraph_getSensorByName(graph, "WristLoadCell_R");
            if (ftsR) {
                auto omForceTorque = new OMForceTorque(graph, ftsR->name, 10000);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {false, true, true, false, false, false}));
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
        
        // Add the task space discrepancy observation model
        if (properties->getPropertyBool("observeTaskSpaceDiscrepancy", true)) {
            auto wamIK = actionModel->unwrap<ActionModelIK>();
            if (wamIK) {
                auto omTSDescrL = new OMTaskSpaceDiscrepancy("ContactPoint_L", graph,
                                                             wamIK->getController()->getGraph());
                fullState->addPart(OMPartial::fromMask(omTSDescrL, {false, true, true}));
                auto omTSDescrR = new OMTaskSpaceDiscrepancy("ContactPoint_R", graph,
                                                             wamIK->getController()->getGraph());
                fullState->addPart(OMPartial::fromMask(omTSDescrR, {false, true, true}));
            }
            else {
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        return fullState.release();
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Box", new PPDBoxExtents(0, true, true, false)); // the Box body has only 1 shape
        manager->addParam("Box", new PPDMassProperties());
        manager->addParam("Box", new PPDMaterialProperties());
        manager->addParam("Table", new PPDMassProperties());
        manager->addParam("Table", new PPDMaterialProperties());
    }
    
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSBoxFlipping(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* box = RcsGraph_getBodyByName(graph, "Box");
        RCHECK(box);
        return new ForceDisturber(box, box);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set camera above plate
        RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
        RCHECK(table);
        std::string cameraView = "tableView";
        properties->getProperty(cameraView, "tableView");
        
        // The camera center is 10cm above the the plate's center
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, table->A_BI->org);
        cameraCenter[0] += 0.0;
        
        // The camera location - not specified yet
        double cameraLocation[3];
        Vec3d_setZero(cameraLocation);
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        if (cameraView == "tableView") {
            RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
            RCHECK(table);
            
            // Rotate to world frame
            Vec3d_transformSelf(cameraLocation, table->A_BI);
            
            // Move the camera
            cameraLocation[0] = table->A_BI->org[0] - 1.8;
            cameraLocation[1] = table->A_BI->org[1];
            cameraLocation[2] = table->A_BI->org[2] + 1.3;
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
                string_format("left cp Y Z | Yd Zd:  [% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f] m/s",
                              obs->ele[omLeftLin.pos], obs->ele[omLeftLin.pos + 1],
                              obs->ele[sd + omLeftLin.vel], obs->ele[sd + omLeftLin.vel + 1]));
            linesOut.emplace_back(
                string_format("right cp Y Z | Yd Zd: [% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f] m/s",
                              obs->ele[omLeftLin.pos + 2], obs->ele[omLeftLin.pos + 3],
                              obs->ele[sd + omLeftLin.vel + 2], obs->ele[sd + omLeftLin.vel + 3]));
        }
        if (!omLeftLin) {
            omLeftLin = observationModel->findOffsets<OMBodyStateLinearPositions>(); // there are two, we find the first
            if (omLeftLin) {
                linesOut.emplace_back(
                    string_format("left cp Y Z:         [% 1.3f,% 1.3f] m",
                                  obs->ele[omLeftLin.pos], obs->ele[omLeftLin.pos + 1]));
                linesOut.emplace_back(
                    string_format("right cp Y Z:        [% 1.3f,% 1.3f] m",
                                  obs->ele[omLeftLin.pos + 2], obs->ele[omLeftLin.pos + 3]));
            }
        }
        
        
        auto omBoxAng = observationModel->findOffsets<OMBodyStateAngular>(); // assuming there is only the box one
        if (!omBoxAng) {
            // If OMBodyStateAngularPositions was used
            omBoxAng = observationModel->findOffsets<OMBodyStateAngularPositions>(); // assuming there is only the box one
        }
        if (omBoxAng && omLeftLin) {
            linesOut.emplace_back(
                string_format("box absolute Y Z | A: [% 1.3f,% 1.3f | % 3.1f] m | deg",
                              obs->ele[omLeftLin.pos + 2], obs->ele[omLeftLin.pos + 3],
                              obs->ele[omBoxAng.pos]*180/M_PI));
        }
        
        
        auto omFTS = observationModel->findOffsets<OMForceTorque>();
        if (omFTS) {
            linesOut.emplace_back(
                string_format("forces left  Y Z:     [% 3.1f, % 3.1f] N",
                              obs->ele[omFTS.pos], obs->ele[omFTS.pos + 1]));
            linesOut.emplace_back(
                string_format("forces right Y Z:     [% 3.1f, % 3.1f] N",
                              obs->ele[omFTS.pos + 2], obs->ele[omFTS.pos + 3]));
        }
        
        auto omColl = observationModel->findOffsets<OMCollisionCost>();
        auto omCollPred = observationModel->findOffsets<OMCollisionCostPrediction>();
        if (omColl && omCollPred) {
            linesOut.emplace_back(
                string_format("coll cost:              %3.2f            pred coll cost: %3.2f",
                              obs->ele[omColl.pos], obs->ele[omCollPred.pos]));
            
        }
        else if (omColl) {
            linesOut.emplace_back(string_format("coll cost:              %3.2f", obs->ele[omColl.pos]));
        }
        else if (omCollPred) {
            linesOut.emplace_back(string_format("pred coll cost:         %3.2f", obs->ele[omCollPred.pos]));
        }
        
        auto omMI = observationModel->findOffsets<OMManipulabilityIndex>();
        if (omMI) {
            linesOut.emplace_back(string_format("manip idx:             %1.3f", obs->ele[omMI.pos]));
        }
        
        auto omGD = observationModel->findOffsets<OMDynamicalSystemGoalDistance>();
        if (omGD) {
            linesOut.emplace_back(
                string_format("goal distance:      [% 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f] m",
                              obs->ele[omGD.pos], obs->ele[omGD.pos + 1], obs->ele[omGD.pos + 2],
                              obs->ele[omGD.pos + 3], obs->ele[omGD.pos + 4]));
        }
        
        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD) {
            linesOut.emplace_back(
                string_format("ts delta left Y Z:    [% 1.3f,% 1.3f] m",
                              obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1]));
            linesOut.emplace_back(
                string_format("ts delta right Y Z:   [% 1.3f,% 1.3f] m",
                              obs->ele[omTSD.pos + 2], obs->ele[omTSD.pos + 3]));
        }
        
        std::stringstream ss;
        ss << "actions:              [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++) {
            ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 3) {
                ss << "\n                      ";
            }
        }
        ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* box_bpi = physicsManager->getBodyInfo("Box");
            BodyParamInfo* table_bpi = physicsManager->getBodyInfo("Table");
            
            linesOut.emplace_back(
                string_format("box width:  %1.3f m           box length: %1.3f m",
                              box_bpi->body->shape[0]->extents[0], box_bpi->body->shape[0]->extents[1]));
            linesOut.emplace_back(
                string_format("box mass:   %1.2f kg      box frict coeff: %1.3f  ",
                              box_bpi->body->m, box_bpi->material.getFrictionCoefficient()));
            linesOut.emplace_back(
                string_format("table mass: %1.2f kg    table frict coeff: %1.3f  ",
                              table_bpi->body->m, table_bpi->material.getFrictionCoefficient()));
        }
    }
    
};

// Register
static ExperimentConfigRegistration<ECBoxFlipping> RegBoxFlipping("BoxFlipping");

}

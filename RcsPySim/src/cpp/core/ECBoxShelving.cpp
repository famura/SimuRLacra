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
#include "initState/ISSBoxShelving.h"
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
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_Vec3d.h>
#include <TaskPosition3D.h>
#include <TaskVelocity1D.h>
#include <TaskOmega1D.h>
#include <TaskEuler3D.h>
#include <TaskEuler1D.h>
#include <TaskPositionForce1D.h>
#include <TaskFactory.h>
#include <CompositeTask.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <memory>
#include <iomanip>

namespace Rcs
{
class ECBoxShelving : public ExperimentConfig
{
protected:
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* leftGrasp = RcsGraph_getBodyByName(graph, "PowerGrasp_L");
        RCHECK(leftGrasp);
        
        // Get reference frames (for now identical for all tasks)
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
            refBody = box;
            refFrame = box;
        }
        else if (refFrameType == "upperGoal") {
            RcsBody* goalUpperShelve = RcsGraph_getBodyByName(graph, "GoalUpperShelve");
            RCHECK(goalUpperShelve);
            refBody = goalUpperShelve;
            refFrame = goalUpperShelve;
        }
        else {
            std::ostringstream os;
            os << "Unsupported reference frame type: " << refFrame;
            throw std::invalid_argument(os.str());
        }
        
        // Initialize action model and tasks
        std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
        std::vector<std::unique_ptr<DynamicalSystem>> tasks;
        
        // Control effector positions and orientation
        if (properties->getPropertyBool("positionTasks", true)) {
            // Left
            innerAM->addTask(new TaskPosition3D(graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskEuler3D(graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(TaskFactory::createTask(
                R"(<Task name="Hand L Joints" controlVariable="Joints" jnts="fing1-knuck1_L tip1-fing1_L fing2-knuck2_L tip2-fing2_L fing3-knuck3_L tip3-fing3_L knuck1-base_L" tmc="0.1" vmax="1000" active="untrue"/>)",
                graph)
            );
            
            // Obtain task data
            unsigned int i = 0;
            std::vector<unsigned int> taskDimsLeft{3, 3, 3, 3, 3, 7};
            std::vector<unsigned int> offsetsLeft{0, 0, 0, 3, 3, 6};
            auto& tsLeft = properties->getChildList("tasksLeft");
            for (auto tsk : tsLeft) {
                DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                i++;
            }
        }
            // Control effector velocity and orientation
        else {
            // Left
            innerAM->addTask(new TaskVelocity1D("Xd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Yd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Ad", graph, leftGrasp, nullptr, nullptr));
            innerAM->addTask(new TaskOmega1D("Bd", graph, leftGrasp, nullptr, nullptr));
            innerAM->addTask(TaskFactory::createTask(
                R"(<Task name="Hand L Joints" controlVariable="Joints" jnts="fing1-knuck1_L tip1-fing1_L fing2-knuck2_L tip2-fing2_L fing3-knuck3_L tip3-fing3_L knuck1-base_L" tmc="0.1" vmax="1000" active="untrue"/>)",
                graph)
            );
            
            // Obtain task data
            unsigned int i = 0;
            std::vector<unsigned int> taskDimsLeft;
            std::vector<unsigned int> offsetsLeft;
            if (properties->getPropertyBool("bidirectionalMPs", true)) {
                taskDimsLeft = {1, 1, 1, 1, 1, 7};
                offsetsLeft = {0, 1, 2, 3, 4, 5};
            }
            else {
                taskDimsLeft = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7};
                offsetsLeft = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
            }
            auto& tsLeft = properties->getChildList("tasksLeft");
            for (auto tsk : tsLeft) {
                DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
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
        
        // Get the method how to combine the movement primitives / tasks given their activation
        std::string taskCombinationMethod = "mean";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(taskCombinationMethod);
        
        return new AMDynamicalSystemActivation(innerAM.release(), taskRel, tcm);
    }
    
    virtual ObservationModel* createObservationModel()
    {
        // Observe effector positions (and velocities)
        std::unique_ptr<OMCombined> fullState(new OMCombined());
        
        if (properties->getPropertyBool("observeVelocities", true)) {
            auto omLeftLin = new OMBodyStateLinear(graph, "PowerGrasp_L"); // in world coordinates
            omLeftLin->setMinState({-0.2, -0.3, 0.7});  // [m]
            omLeftLin->setMaxState({2.1, 0.6, 1.8});  // [m]
            omLeftLin->setMaxVelocity(3.); // [m/s]
            fullState->addPart(omLeftLin);
        }
        else {
            auto omLeftLin = new OMBodyStateLinearPositions(graph, "PowerGrasp_L"); // in world coordinates
            omLeftLin->setMinState({-0.2, -0.3, 0.7});  // [m]
            omLeftLin->setMaxState({2.1, 0.6, 1.8});  // [m]
            fullState->addPart(omLeftLin);
        }
        
        // Observe box positions (and velocities)
        if (properties->getPropertyBool("observeVelocities", true)) {
            auto omBoxLin = new OMBodyStateLinear(graph, "Box", "GoalUpperShelve",
                                                  "GoalUpperShelve");  // in relative coordinates
            omBoxLin->setMinState({-0.2, -0.4, -0.4});  // [m]
            omBoxLin->setMaxState({1.8, 0.4, 1.0});  // [m]
            omBoxLin->setMaxVelocity(5.); // [m/s]
            fullState->addPart(omBoxLin);
        }
        else {
            auto omBoxLin = new OMBodyStateLinearPositions(graph, "Box", "GoalUpperShelve",
                                                           "GoalUpperShelve");  // in relative coordinates
            omBoxLin->setMinState({-0.2, -0.4, -0.4});  // [m]
            omBoxLin->setMaxState({1.8, 0.4, 1.0});  // [m]
            fullState->addPart(omBoxLin);
        }
        
        // Observe box orientation in the world frame
        if (properties->getPropertyBool("observeVelocities", true)) {
            auto omBoxAng = new OMBodyStateAngular(graph, "Box");
            omBoxAng->setMaxVelocity(RCS_DEG2RAD(360)); // [rad/s]
            fullState->addPart(omBoxAng);
        }
        else {
            auto omBoxAng = new OMBodyStateAngularPositions(graph, "Box");
            fullState->addPart(omBoxAng);
        }
        
        // Add goal distances
        if (properties->getPropertyBool("observeDynamicalSystemGoalDistance", false)) {
            auto amAct = actionModel->unwrap<AMDynamicalSystemActivation>();
            RCHECK(amAct);
            fullState->addPart(new OMDynamicalSystemGoalDistance(amAct));
        }
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true)) {
            RcsSensor* ftsL = RcsGraph_getSensorByName(graph, "WristLoadCellLBR_L");
            if (ftsL) {
                auto omForceTorque = new OMForceTorque(graph, ftsL->name, 1200);
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
                auto omTSDescr = new OMTaskSpaceDiscrepancy("PowerGrasp_L", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescr);
            }
            else {
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        return fullState.release();
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Box", new PPDMassProperties());
        manager->addParam("Box", new PPDMaterialProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSBoxShelving(graph);
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
        std::string cameraView = "overLeftShoulder";
        properties->getProperty(cameraView, "overLeftShoulder");
        
        // The camera center is 10cm above the the plate's center
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, table->A_BI->org);
        cameraCenter[2] += 0.1;
        
        // The camera location - not specified yet
        double cameraLocation[3];
        Vec3d_setZero(cameraLocation);
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        if (cameraView == "overLeftShoulder") {
            RcsBody* leftShoulder = RcsGraph_getBodyByName(graph, "lbr_link_0_L");
            // Shift camera behind the left shoulder
            cameraLocation[0] = leftShoulder->A_BI->org[0] - 2.2;
            cameraLocation[1] = leftShoulder->A_BI->org[1] + 1.2;
            cameraLocation[2] = leftShoulder->A_BI->org[2] + 0.5;
            
            // Rotate to world frame
            Vec3d_transformSelf(cameraLocation, table->A_BI);
            
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
                string_format("left hand pg: [% 1.3f,% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f,% 1.3f] m/s",
                              obs->ele[omLeftLin.pos], obs->ele[omLeftLin.pos + 1], obs->ele[omLeftLin.pos + 2],
                              obs->ele[sd + omLeftLin.vel], obs->ele[sd + omLeftLin.vel + 1],
                              obs->ele[sd + omLeftLin.vel + 2]));
        }
        
        auto omBoxAng = observationModel->findOffsets<OMBodyStateAngular>(); // assuming there is only the box one
        if (omBoxAng && omLeftLin) {
            linesOut.emplace_back(
                string_format("box relative: [% 1.3f,% 1.3f,% 1.3f] m   [% 3.0f,% 3.0f,% 3.0f] deg",
                              obs->ele[omLeftLin.pos + 3], obs->ele[omLeftLin.pos + 4], obs->ele[omLeftLin.pos + 5],
                              obs->ele[omBoxAng.pos]*180/M_PI, obs->ele[omBoxAng.pos + 1]*180/M_PI,
                              obs->ele[omBoxAng.pos + 2]*180/M_PI));
        }
        else if (omLeftLin) {
            linesOut.emplace_back(
                string_format("box relative: [% 1.3f,% 1.3f,% 1.3f] m",
                              obs->ele[omLeftLin.pos + 3], obs->ele[omLeftLin.pos + 4], obs->ele[omLeftLin.pos + 5]));
        }
        
        auto omFTS = observationModel->findOffsets<OMForceTorque>();
        if (omFTS) {
            linesOut.emplace_back(
                string_format("forces:       [% 3.1f, % 3.1f, % 3.1f] N ", obs->ele[omFTS.pos], obs->ele[omFTS.pos + 1],
                              obs->ele[omFTS.pos + 2]));
        }
        
        auto omColl = observationModel->findOffsets<OMCollisionCost>();
        auto omCollPred = observationModel->findOffsets<OMCollisionCostPrediction>();
        if (omColl && omCollPred) {
            linesOut.emplace_back(
                string_format("coll cost:      %3.2f                     pred coll cost: %3.2f", obs->ele[omColl.pos],
                              obs->ele[omCollPred.pos]));
            
        }
        else if (omColl) {
            linesOut.emplace_back(string_format("coll cost:      %3.2f", obs->ele[omColl.pos]));
        }
        else if (omCollPred) {
            linesOut.emplace_back(string_format("pred coll cost:  %3.2f", obs->ele[omCollPred.pos]));
        }
        
        auto omMI = observationModel->findOffsets<OMManipulabilityIndex>();
        if (omMI) {
            linesOut.emplace_back(string_format("manip idx:      %1.3f", obs->ele[omMI.pos]));
        }
        
        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD) {
            linesOut.emplace_back(
                string_format("ts delta:     [% 1.3f,% 1.3f,% 1.3f] m", obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1],
                              obs->ele[omTSD.pos + 2]));
        }
        
        std::stringstream ss;
        ss << "actions:      [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++) {
            ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6) {
                ss << "\n               ";
            }
        }
        ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
        
        auto castedAM = dynamic_cast<AMDynamicalSystemActivation*>(actionModel);
        if (castedAM) {
            std::stringstream ss;
            ss << "activations:  [";
            for (unsigned int i = 0; i < castedAM->getDim() - 1; i++) {
                ss << std::fixed << std::setprecision(2) << MatNd_get(castedAM->getActivation(), i, 0) << ", ";
                if (i == 6) {
                    ss << "\n               ";
                }
            }
            ss << std::fixed << std::setprecision(2) <<
               MatNd_get(castedAM->getActivation(), castedAM->getDim() - 1, 0) << "]";
            linesOut.emplace_back(string_format(ss.str()));
            
            linesOut.emplace_back(string_format("tcm:           %s", castedAM->getTaskCombinationMethodName()));
        }
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* box_bpi = physicsManager->getBodyInfo("Box");
            
            linesOut.emplace_back(string_format("box mass:      %1.2f kg         box frict coeff: %1.3f ",
                                                box_bpi->body->m, box_bpi->material.getFrictionCoefficient()));
        }
    }
    
};

// Register
static ExperimentConfigRegistration<ECBoxShelving> RegBoxShelving("BoxShelving");
    
}

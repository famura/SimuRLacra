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
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMDynamicalSystemActivation.h"
#include "action/AMIKControllerActivation.h"
#include "action/ActionModelIK.h"
#include "initState/ISSPlanarInsert.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMCombined.h"
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
#include "physics/PPDBodyPosition.h"
#include "physics/ForceDisturber.h"
#include "physics/PPDMaterialProperties.h"
#include "util/string_format.h"

#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <TaskVelocity1D.h>
#include <TaskOmega1D.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Rcs
{

class ECPlanarInsert : public ExperimentConfig
{
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Get the method how to combine the movement primitives / tasks given their activation
        std::string taskCombinationMethod = "unspecified";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(taskCombinationMethod);
        
        // Common for the action models
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
        
        if (actionModelType == "ik_activation") {
            // Create the action model
            auto amIK = new AMIKControllerActivation(graph, tcm);
            std::vector<Task*> tasks;
            
            // Check if the tasks are defined on position or velocity level. Adapt their parameters if desired.
            if (properties->getPropertyBool("positionTasks", false)) {
                throw std::invalid_argument("Position tasks based are not implemented for PlanarInsert!");
            }
            else {
                tasks.emplace_back(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
                tasks.emplace_back(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
                tasks.emplace_back(new TaskOmega1D("Bd", graph, effector, nullptr, nullptr));
                tasks[0]->resetParameter(Task::Parameters(-0.5, 0.5, 1.0, "X Velocity [m/s]"));
                tasks[1]->resetParameter(Task::Parameters(-0.5, 0.5, 1.0, "Z Velocity [m/s]"));
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
            // Obtain the inner action model
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            innerAM->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
            innerAM->addTask(new TaskOmega1D("Bd", graph, effector, nullptr, nullptr));
            
            // Obtain task data
            unsigned int i = 0;
            std::vector<unsigned int> offsets{0, 0, 1, 1, 2, 2}; // depends on the order of the MPs coming from python
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
            auto& taskSpec = properties->getChildList("tasks");
            for (auto tsk : taskSpec) {
                // Positive and negative linear velocity tasks separately
                DynamicalSystem* ds = DynamicalSystem::create(tsk, 1);
                tasks.emplace_back(new DSSlice(ds, offsets[i], 1));
                i++;
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
        
        // Observe effector position
        auto omLin = new OMBodyStateLinear(graph, "Effector", "GroundPlane"); // Base center is above ground level
        omLin->setMinState(-1.7); // [m]
        omLin->setMaxState(1.7); // [m]
        omLin->setMaxVelocity(5.); // [m/s]
        fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));  // mask out y axis
        
        auto omAng = new OMBodyStateAngular(graph, "Effector", "GroundPlane"); // Base center is above ground level
        omAng->setMaxVelocity(20.); // [rad/s]
        fullState->addPart(OMPartial::fromMask(omAng, {false, true, false}));  // only y axis
        
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        if (actionModelType == "ds_activation") {
            if (properties->getPropertyBool("observeDynamicalSystemGoalDistance", false)) {
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
                auto omForceTorque = new OMForceTorque(graph, fts->name, 300);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, false, true, false, false, false}));
            }
        }
        
        // Add current collision cost
        if (properties->getPropertyBool("observeCollisionCost", false) & (collisionMdl != nullptr)) {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        // Add predicted collision cost
        if (properties->getPropertyBool("observePredictedCollisionCost", false) && collisionMdl != nullptr) {
            // Get horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add collision model
            auto omCollisionCost = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, 50);
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
        manager->addParam("Link4", new PPDMassProperties());
        manager->addParam("Effector", new PPDMaterialProperties());
        manager->addParam("UpperWall", new PPDBodyPosition(false, false, true)); // only z position
        manager->addParam("UpperWall", new PPDMaterialProperties());
    }
    
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSPlanarInsert(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
        return new ForceDisturber(effector, effector);
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
        std::vector<std::string>& linesOut, double currentTime, const MatNd* obs, const MatNd* currentAction,
        PhysicsBase* simulator, PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != nullptr) {
            simName = simulator->getClassName();
        }
        
        linesOut.emplace_back(
            string_format("physics engine: %s                            sim time: %2.3f s", simName, currentTime));
        
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
        auto omAng = observationModel->findOffsets<OMBodyStateAngular>();
        if (omLin && omAng) {
            
            linesOut.emplace_back(string_format("end-eff pos:   [% 1.3f,% 1.3f,% 1.3f] m, m, deg",
                                                obs->ele[omLin.pos], obs->ele[omLin.pos + 1],
                                                RCS_RAD2DEG(obs->ele[omAng.pos])));
            
            linesOut.emplace_back(string_format("end-eff vel:   [% 1.3f,% 1.3f,% 1.3f] m/s, m/s, deg/s",
                                                obs->ele[sd + omLin.vel],
                                                obs->ele[sd + omLin.vel + 1],
                                                RCS_RAD2DEG(obs->ele[sd + omAng.vel])));
        }
        
        auto omFT = observationModel->findOffsets<OMForceTorque>();
        if (omFT) {
            linesOut.emplace_back(
                string_format("forces:        [% 3.1f,% 3.1f] N", obs->ele[omFT.pos], obs->ele[omFT.pos + 1]));
        }
        
        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD) {
            linesOut.emplace_back(
                string_format("ts delta:      [% 1.3f,% 1.3f] m", obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1]));
        }
        
        std::stringstream ss;
        ss << "actions:       [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++) {
            ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6) {
                ss << "\n               ";
            }
        }
        ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
        
        auto castedAM = dynamic_cast<AMDynamicalSystemActivation*>(actionModel);
        if (castedAM) {
            std::stringstream ss;
            ss << "activations:   [";
            for (unsigned int i = 0; i < castedAM->getDim() - 1; i++) {
                ss << std::fixed << std::setprecision(3) << MatNd_get(castedAM->getActivation(), i, 0) << ", ";
                if (i == 6) {
                    ss << "\n               ";
                }
            }
            ss << std::fixed << std::setprecision(3) <<
               MatNd_get(castedAM->getActivation(), castedAM->getDim() - 1, 0) << "]";
            linesOut.emplace_back(string_format(ss.str()));
            
            linesOut.emplace_back(string_format("tcm:            %s", castedAM->getTaskCombinationMethodName()));
        }
        
        if (forceDisturber) {
            const double* distForce = forceDisturber->getLastForce();
            linesOut.emplace_back(
                string_format("disturbances:  [% 3.1f,% 3.1f,% 3.1f] N", distForce[0], distForce[1], distForce[2]));
        }
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* link1_bpi = physicsManager->getBodyInfo("Link1");
            BodyParamInfo* link2_bpi = physicsManager->getBodyInfo("Link2");
            BodyParamInfo* link3_bpi = physicsManager->getBodyInfo("Link3");
            BodyParamInfo* link4_bpi = physicsManager->getBodyInfo("Link4");
            BodyParamInfo* uWall_bpi = physicsManager->getBodyInfo("UpperWall");
            BodyParamInfo* lWall_bpi = physicsManager->getBodyInfo("LowerWall");
            BodyParamInfo* eff_bpi = physicsManager->getBodyInfo("Effector");
            
            linesOut.emplace_back(
                string_format("link masses:   [%1.2f, %1.2f, %1.2f, %1.2f] kg      wall Z pos: %1.3f m",
                              link1_bpi->body->m, link2_bpi->body->m, link3_bpi->body->m, link4_bpi->body->m,
                              uWall_bpi->body->A_BP->org[2]));
            linesOut.emplace_back(string_format("wall friction: [%1.3f, %1.3f] l/u        effector friction: %1.3f",
                                                lWall_bpi->material.getFrictionCoefficient(),
                                                uWall_bpi->material.getFrictionCoefficient(),
                                                eff_bpi->material.getFrictionCoefficient()));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECPlanarInsert> RegPlanarInsert("PlanarInsert");
    
}

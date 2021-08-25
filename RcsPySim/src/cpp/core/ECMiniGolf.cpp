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
#include "action/AMJointControlPosition.h"
#include "initState/ISSMiniGolf.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "observation/OMForceTorque.h"
#include "observation/OMPartial.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDBodyOrientation.h"
#include "physics/PPDBodyPosition.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/PPDSphereRadius.h"
#include "util/string_format.h"

#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <TaskDistance1D.h>
#include <TaskEuler1D.h>
#include <TaskFactory.h>
#include <TaskPosition1D.h>
#include <TaskVelocity1D.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace Rcs
{

class ECMiniGolf : public ExperimentConfig
{
    
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Common for the action models
        RcsBody* ground = RcsGraph_getBodyByName(graph, "Ground");
        RCHECK(ground);
        RcsBody* club = RcsGraph_getBodyByName(graph, "Club");
        RCHECK(club);
        RcsBody* clubTip = RcsGraph_getBodyByName(graph, "ClubTip");
        RCHECK(clubTip);
        RcsBody* ball = RcsGraph_getBodyByName(graph, "Ball");
        RCHECK(ball);
        
        if (actionModelType == "joint_pos") {
            return new AMJointControlPosition(graph);
        }
        
        else if (actionModelType == "ik") {
            // Create the action model. Every but the x tasks have been fixed tasks originally, but now are constant
            // outputs on the policy. This way, we can use the same policy structure in the pre-strike ControlPolicy
            // on the real robot.
            auto amIK = new AMIKGeneric(graph);
            if (properties->getPropertyBool("positionTasks", true)) {
                if (properties->getPropertyBool("relativeZdTask", true)) {
                    // Driving
                    auto tmpTask = new TaskVelocity1D("Zd", graph, ball, clubTip, nullptr);
                    tmpTask->resetParameter(Task::Parameters(-100.0, 100.0, 1.0, "Z Velocity [m/s]"));
                    amIK->addTask(tmpTask);
                    // Centering
                    amIK->addTask(new TaskPosition1D("Y", graph, ball, clubTip, nullptr));
                    amIK->addTask(new TaskDistance1D(graph, club, ground, 2));
                    amIK->addTask(TaskFactory::createTask(
                        R"(<Task name="ClubTip_Polar" controlVariable="POLAR" effector="ClubTip"  active="true" />)",
                        graph)
                    );
                }
                else {
                    // Driving
                    amIK->addTask(new TaskPosition1D("X", graph, clubTip, nullptr, nullptr));
                    // Centering
//                    amIK->addTask(new TaskPosition1D("X", graph, ball, clubTip, nullptr));
                    amIK->addTask(new TaskPosition1D("Y", graph, ball, clubTip, nullptr));
                    amIK->addTask(new TaskDistance1D(graph, club, ground, 2));
                    amIK->addTask(TaskFactory::createTask(
                        R"(<Task name="ClubTip_Polar" controlVariable="POLAR" effector="ClubTip"  active="true" />)",
                        graph)
                    );
                    /*
                    amIK->addTask(new TaskPosition1D("X", graph, clubTip, refBody, refFrame));
                    amIK->addTask(new TaskPosition1D("Y", graph, ball, clubTip, ground));
                    amIK->addTask(new TaskDistance1D(graph, club, ground, 2));
                    amIK->addTask(new TaskEuler1D("C", graph, clubTip, nullptr, ground));
                     */
                }
            }
            
            else {
                throw std::invalid_argument("Velocity tasks are not implemented for AMIKGeneric in this environment.");
            }
            
            // Incorporate collision costs into IK
            if (properties->getPropertyBool("collisionAvoidanceIK", false)) {
                REXEC(4) {
                    std::cout << "IK considers the provided collision model" << std::endl;
                }
                amIK->setupCollisionModel(collisionMdl);
            }
            
            return amIK;
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
        
        // Observe the ball
        if (properties->getPropertyBool("observeVelocities", false)) {
            auto omLinBall = new OMBodyStateLinear(graph, "Ball", nullptr);
            omLinBall->setMinState({-5, -5, 0}); // [m]
            omLinBall->setMaxState({3, 3, 0.1}); // [m]
            omLinBall->setMaxVelocity(10); // [m/s]
            fullState->addPart(omLinBall);
        }
        else {
            auto omLinBall = new OMBodyStateLinearPositions(graph, "Ball", nullptr);
            omLinBall->setMinState({-3, -3, 0}); // [m]
            omLinBall->setMaxState({3, 3, 0.1}); // [m]
            fullState->addPart(omLinBall);
        }
        
        // Observe the club
        if (properties->getPropertyBool("observeVelocities", false)) {
            auto omLinClub = new OMBodyStateLinear(graph, "ClubTip", nullptr);
            omLinClub->setMinState({-3, -3, 0}); // [m]
            omLinClub->setMaxState({3, 3, 2}); // [m]
            omLinClub->setMaxVelocity(5); // [m/s]
            fullState->addPart(omLinClub);
        }
        else {
            auto omLinClub = new OMBodyStateLinearPositions(graph, "ClubTip", nullptr);
            omLinClub->setMinState({-3, -3, 0}); // [m]
            omLinClub->setMaxState({3, 3, 2}); // [m]
            fullState->addPart(omLinClub);
        }
        
        // Observe the club
        if (properties->getPropertyBool("observeVelocities", false)) {
            auto omAng = new OMBodyStateAngular(graph, "ClubTip", nullptr);
            omAng->setMaxVelocity(20); // [rad/s]
            fullState->addPart(omAng);
        }
        else {
            auto omAng = new OMBodyStateAngularPositions(graph, "ClubTip", nullptr);
            fullState->addPart(omAng);
        }
        
        // Observe the robot's joints
        std::list<std::string> listOfJointNames = {"base-m3", "m3-m4", "m4-m5", "m5-m6", "m6-m7", "m7-m8", "m8-m9"};
        for (std::string jointName : listOfJointNames) {
            fullState->addPart(new OMJointStatePositions(graph, jointName.c_str(), false));
        }
        
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", false)) {
            RcsSensor* fts = RcsGraph_getSensorByName(graph, "WristLoadCellSchunk");
            if (fts) {
                auto omForceTorque = new OMForceTorque(graph, fts->name, 1000);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, true, true, false, false, false}));
            }
        }
        
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Ball", new PPDSphereRadius("Ground"));
        manager->addParam("Ball", new PPDMassProperties());
        manager->addParam("Ball", new PPDMaterialProperties());
        manager->addParam("Club", new PPDMassProperties());
        manager->addParam("Ground", new PPDMaterialProperties());
        manager->addParam("ObstacleLeft", new PPDBodyPosition(true, true, false));
        manager->addParam("ObstacleLeft", new PPDBodyOrientation(false, false, true));
        manager->addParam("ObstacleRight", new PPDBodyPosition(true, true, false));
        manager->addParam("ObstacleRight", new PPDBodyOrientation(false, false, true));
    }
    
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSMiniGolf(graph, properties->getPropertyBool("fixedInitState", false));
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set the camera center
        double cameraCenter[3];
        cameraCenter[0] = 1.5;
        cameraCenter[1] = 0.5;
        cameraCenter[2] = 0.0;
        
        // Set the camera position
        double cameraLocation[3];
        cameraLocation[0] = -1.5;
        cameraLocation[1] = 4.5;
        cameraLocation[2] = 2.8;
        
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
        const char* simName;
        if (simulator != nullptr) {
            simName = simulator->getClassName();
        }
        else {
            simName = "Robot";
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

//        unsigned int sd = observationModel->getStateDim();
        
        auto omLinBall = observationModel->findOffsets<OMBodyStateLinear>(); // finds the first OMBodyStateLinear
        if (omLinBall) {
            linesOut.emplace_back(string_format("ball pos:     [% 1.3f,% 1.3f,% 1.3f] m",
                                                obs->ele[omLinBall.pos],
                                                obs->ele[omLinBall.pos + 1],
                                                obs->ele[omLinBall.pos + 2]));
            linesOut.emplace_back(string_format("club tip pos: [% 1.3f,% 1.3f,% 1.3f] m",
                                                obs->ele[omLinBall.pos + 3],
                                                obs->ele[omLinBall.pos + 4],
                                                obs->ele[omLinBall.pos + 5]));
        }
        
        auto omAng = observationModel->findOffsets<OMBodyStateAngular>();
        if (omAng) {
            linesOut.emplace_back(string_format("club tip ang: [% 1.3f,% 1.3f,% 1.3f] deg",
                                                RCS_RAD2DEG(obs->ele[omAng.pos]),
                                                RCS_RAD2DEG(obs->ele[omAng.pos + 1]),
                                                RCS_RAD2DEG(obs->ele[omAng.pos + 2])));
        }
        
        auto omFT = observationModel->findOffsets<OMForceTorque>();
        if (omFT) {
            linesOut.emplace_back(
                string_format("forces:       [% 3.1f,% 3.1f,% 3.1f] N",
                              obs->ele[omFT.pos], obs->ele[omFT.pos + 1], obs->ele[omFT.pos + 2]));
        }
        
        std::stringstream ss;
        ss << "actions:      [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++) {
            ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6) {
                ss << "\n               ";
            }
        }
        ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* ball_bpi = physicsManager->getBodyInfo("Ball");
            BodyParamInfo* club_bpi = physicsManager->getBodyInfo("Club");
            BodyParamInfo* ground_bpi = physicsManager->getBodyInfo("Ground");
            
            double ballSlip = 0;
            ball_bpi->material.getDouble("slip", ballSlip);
            double groundSlip = 0;
            ground_bpi->material.getDouble("slip", groundSlip);
            
            linesOut.emplace_back(
                string_format("ball mass:             %1.2f kg           club mass: %1.2f kg",
                              ball_bpi->body->m, club_bpi->body->m));
            linesOut.emplace_back(string_format("ball friction:         %1.3f      ground friction: %1.3f",
                                                ball_bpi->material.getFrictionCoefficient(),
                                                ground_bpi->material.getFrictionCoefficient()));
            linesOut.emplace_back(string_format("ball rolling friction: %1.6f         ball slip: %1.5f rad/(Ns)",
                                                ball_bpi->material.getRollingFrictionCoefficient(),
                                                ballSlip));
            linesOut.emplace_back(string_format("ball restitution: %1.3f               ground slip: %1.5f rad/(Ns)",
                                                ball_bpi->material.getRestitution(), groundSlip));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECMiniGolf> RegMiniGolf("MiniGolf");
    
}

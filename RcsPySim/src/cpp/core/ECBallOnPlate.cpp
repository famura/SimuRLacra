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
#include "action/AMJointControlPosition.h"
#include "action/AMIntegrate1stOrder.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMPlateAngPos.h"
#include "action/AMPlatePos5D.h"
#include "initState/ISSBallOnPlate.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMBallPos.h"
#include "observation/OMCombined.h"
#include "observation/OMPartial.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDSphereRadius.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <stdexcept>
#include <cmath>

namespace Rcs
{

class ECBallOnPlate : public ExperimentConfig
{
    double initManipulability;

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "joint_pos";
        properties->getProperty(actionModelType, "actionModelType");
        
        if (actionModelType == "joint_pos") {
            return new AMJointControlPosition(graph);
        }
        else if (actionModelType == "joint_acc") {
            double maxAction = 120*M_PI/180; // [1/s^2]
            properties->getProperty(maxAction, "maxAction");
            return new AMIntegrate2ndOrder(new AMJointControlPosition(graph), maxAction);
        }
        else if (actionModelType == "plate_angpos") {
            return new AMPlateAngPos(graph);
        }
        else if (actionModelType == "plate_angvel") {
            double maxAction = 120*M_PI/180; // [1/s]
            properties->getProperty(maxAction, "maxAction");
            return new AMIntegrate1stOrder(new AMPlateAngPos(graph), maxAction);
        }
        else if (actionModelType == "plate_angacc") {
            double maxAction = 120*M_PI/180; // [1/s^2]
            properties->getProperty(maxAction, "maxAction");
            return new AMIntegrate2ndOrder(new AMPlateAngPos(graph), maxAction);
        }
        else if (actionModelType == "plate_acc5d") {
            MatNd* maxAction;
            MatNd_fromStack(maxAction, 5, 1);
            // Use different max action for linear/angular
            double max_lin = 0.5; // [m/s^2]
            double max_ang = 120*M_PI/180; // [1/s^2]
            
            maxAction->ele[0] = max_lin;
            maxAction->ele[1] = max_lin;
            maxAction->ele[2] = max_lin;
            maxAction->ele[3] = max_ang;
            maxAction->ele[4] = max_ang;
            
            return new AMIntegrate2ndOrder(new AMPlatePos5D(graph), maxAction);
        }
        else {
            std::ostringstream os;
            os << "Unsupported action model type: " << actionModelType;
            throw std::invalid_argument(os.str());
        }
    }
    
    virtual ObservationModel* createObservationModel()
    {
        RcsBody* plate = RcsGraph_getBodyByName(graph, "Plate");
        RcsBody* plateOrigMarker = RcsGraph_getBodyByName(graph, "PlateOriginMarker");
        RCHECK_MSG(plateOrigMarker, "PlateOriginMarker is missing, please update your xml.");
        // Set origin marker position
        HTr_copyOrRecreate(&plateOrigMarker->A_BP, plate->A_BI);
        // Select observation model
        
        bool ballObsAngular = properties->getPropertyBool("ballObsAngular");
//        bool ballObsInInertialFrame = properties->getPropertyBool(
//                "ballObsInInertialFrame");
        
        std::string actionModelType = "joint_pos";
        properties->getProperty(actionModelType, "actionModelType");
        bool havePlateLinPos = actionModelType == "plate_acc5d";
        
        auto fullState = new OMCombined();
        
        // Add plate linear position if desired
        if (havePlateLinPos) {
            auto plin = new OMBodyStateLinear(graph, "Plate", "PlateOriginMarker");
            plin->setMinState(-0.1);
            plin->setMaxState(0.1);
            plin->setMaxVelocity(2.0);
            fullState->addPart(plin);
        }
        
        // Plate angular position
        auto pang = new OMBodyStateAngular(graph, "Plate", "PlateOriginMarker");
        pang->setMinState(-45*M_PI/180);
        pang->setMaxState(45*M_PI/180);
        pang->setMaxVelocity(2*360*M_PI/180);
        // Mask out Z since it's fixed
        fullState->addPart(OMPartial::fromMask(pang, {true, true, false}));
        
        // Ball linear state
        fullState->addPart(new OMBallPos(graph));
        
        if (ballObsAngular) {
            // Add angular ball state
            fullState->addPart(new OMBodyStateAngular(graph, "Ball", "Plate"));
        }
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Ball", new PPDSphereRadius("Plate"));
        manager->addParam("Ball", new PPDMassProperties());
        manager->addParam("Ball", new PPDMaterialProperties());
        manager->addParam("Plate", new PPDMaterialProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSBallOnPlate(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* ball = RcsGraph_getBodyByName(graph, "Ball");
        RcsBody* plate = RcsGraph_getBodyByName(graph, "Plate");
        return new ForceDisturber(ball, plate);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
        initManipulability = 0;
        // Compute initial manipulability
        auto amIK = actionModel->unwrap<ActionModelIK>();
        if (amIK != NULL) {
            initManipulability = amIK->getController()->computeManipulabilityCost();
        }

#ifdef GRAPHICS_AVAILABLE
        // Set camera above plate
        RcsBody* plate = RcsGraph_getBodyByName(graph, "Plate");
        RCHECK(plate);
        std::string cameraView = "side";
        properties->getProperty(cameraView, "cameraView");
        
        // The camera center is the plate center
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, plate->A_BI->org);
        
        // The camera location - not specified yet
        double cameraLocation[3];
        Vec3d_setZero(cameraLocation);
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        if (cameraView == "topdown") {
            // The camera looks down vertically
            Vec3d_copy(cameraLocation, cameraCenter);
            // The distance between camera and plate is configurable
            double cameraToPlateDistanceZ = 1.6;
            properties->getProperty(cameraToPlateDistanceZ, "cameraToPlateDistance");
            cameraLocation[2] += cameraToPlateDistanceZ;
            // The plate's y vector should point up on the view
            Vec3d_setUnitVector(cameraUp, 1);
            Vec3d_transRotateSelf(cameraUp, plate->A_BI->rot);
            
        }
        else if (cameraView == "side") {
            // Make camera center a bit lower, so we use the screen better
            cameraCenter[2] -= 0.4;
            // Shift camera pos in x dir
            cameraLocation[0] = 2.9;
            cameraLocation[2] = 1.3;
            
            // Rotate to world frame
            Vec3d_transformSelf(cameraLocation, plate->A_BI);
            
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
    
    void getHUDText(
        std::vector<std::string>& linesOut, double currentTime, const MatNd* obs, const MatNd* currentAction,
        PhysicsBase* simulator, PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != NULL) {
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
            "plate angles:     [% 3.2f,% 3.2f] deg",
            RCS_RAD2DEG(obs->ele[0]), RCS_RAD2DEG(obs->ele[1])));
        
        const double* distForce = forceDisturber->getLastForce();
        linesOut.emplace_back(string_format(
            "disturbances:   [% 3.1f,% 3.1f,% 3.1f] N",
            distForce[0], distForce[1], distForce[2]));
        
        if (physicsManager != NULL) {
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
        
        // Compute manipulability cost if available
        auto amIK = actionModel->unwrap<ActionModelIK>();
        if (amIK != NULL) {
            double manipulability = amIK->getController()->computeManipulabilityCost()/initManipulability;
            linesOut.emplace_back(string_format(
                "manipulability: %10.8f ",
                manipulability));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECBallOnPlate> RegBallOnPlate("BallOnPlate");

}

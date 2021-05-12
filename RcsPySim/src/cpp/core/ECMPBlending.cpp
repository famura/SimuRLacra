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
#include "action/AMIKControllerActivation.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMDynamicalSystemActivation.h"
#include "action/ActionModelIK.h"
#include "initState/ISSMPBlending.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMCombined.h"
#include "observation/OMPartial.h"
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

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Rcs
{

class ECMPblending : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Get the method how to combine the movement primitives / tasks given their activation
        std::string taskCombinationMethod = "unspecified";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMDynamicalSystemActivation::checkTaskCombinationMethod(taskCombinationMethod);
        
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
        
        if (actionModelType == "ds_activation") {
            // Obtain the inner action model
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            
            // Check if the MPs are defined on position or velocity level
            if (properties->getPropertyBool("positionTasks", true)) {
                innerAM->addTask(new TaskPosition1D("X", graph, effector, nullptr, nullptr));
                innerAM->addTask(new TaskPosition1D("Y", graph, effector, nullptr, nullptr));
            }
            else {
                innerAM->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
                innerAM->addTask(new TaskVelocity1D("Yd", graph, effector, nullptr, nullptr));
            }
            
            // Obtain the task data
            auto& taskSpec = properties->getChildList("tasks");
            if (taskSpec.empty()) {
                throw std::invalid_argument("No tasks specified!");
            }
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
            for (auto ts : taskSpec) {
                // All tasks cover the x and y coordinate
                tasks.emplace_back(DynamicalSystem::create(ts, innerAM->getDim()));
            }
            
            // Setup task-based action model
            std::vector<DynamicalSystem*> taskRel;
            for (auto& task : tasks) {
                taskRel.push_back(task.release());
            }
            
            // Create the action model
            return new AMDynamicalSystemActivation(innerAM.release(), taskRel, tcm);
        }
        
        else if (actionModelType == "ik_activation") {
            // Create the action model
            auto amIK = new AMIKControllerActivation(graph, tcm);
            std::vector<Task*> tasks;
    
            // Check if the tasks are defined on position or velocity level. Adapt their parameters if desired.
            if (properties->getPropertyBool("positionTasks", true)) {
                RcsBody* goalLL = RcsGraph_getBodyByName(graph, "GoalLL");
                RcsBody* goalUL = RcsGraph_getBodyByName(graph, "GoalUL");
                RcsBody* goalLR = RcsGraph_getBodyByName(graph, "GoalLR");
                RcsBody* goalUR = RcsGraph_getBodyByName(graph, "GoalUR");
                RCHECK(goalLL);
                RCHECK(goalUL);
                RCHECK(goalLR);
                RCHECK(goalUR);
                int i = 0;
                
                tasks.emplace_back(new TaskPosition3D(graph, effector, goalLL, nullptr));
                tasks.emplace_back(new TaskPosition3D(graph, effector, goalUL, nullptr));
                tasks.emplace_back(new TaskPosition3D(graph, effector, goalLR, nullptr));
                tasks.emplace_back(new TaskPosition3D(graph, effector, goalUR, nullptr));
                for (auto task : tasks) {
                    std::stringstream taskName;
                    taskName << " Position " << i++ << " [m]";
                    task->resetParameter(
                        Task::Parameters(-1.2, 1.2, 1.0, static_cast<std::string>("X") + taskName.str()));
                    task->addParameter(
                        Task::Parameters(-1.2, 1.2, 1.0, static_cast<std::string>("Y") + taskName.str()));
                    task->addParameter(Task::Parameters(0.1, 0.2, 1.0, static_cast<std::string>("Z") + taskName.str()));
                }
            }
            else {
                throw std::invalid_argument("The combination of velocity-based tasks and AMIKControllerActivation is"
                                            "not supported!");
            }
    
            // Add the tasks
            for (auto t : tasks) { amIK->addTask(t); }
    
            // Set the tasks' desired states
            std::vector<PropertySource*> taskSpec = properties->getChildList("taskSpecIK");
            amIK->setXdesFromTaskSpec(taskSpec);
    
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
        
        // Observe effector position
        auto omLin = new OMBodyStateLinear(graph, "Effector", "GroundPlane");
        fullState->addPart(OMPartial::fromMask(omLin, {true, true, false}));  // mask out z axis
        
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Effector", new PPDMassProperties());  // not necessary
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSMPBlending(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* eff = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(eff);
        return new ForceDisturber(eff, NULL);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set camera next to base
        RcsBody* base = RcsGraph_getBodyByName(graph, "Effector");
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, base->A_BI->org);
        
        // Set the camera position
        double cameraLocation[3];
        cameraLocation[0] = 1.5;
        cameraLocation[1] = -2.5;
        cameraLocation[2] = 4.;
        
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
        
        unsigned int sd = observationModel->getStateDim();
        
        linesOut.emplace_back(
            string_format("end-eff pos:   [% 1.3f,% 1.3f] m  end-eff vel:   [% 1.2f,% 1.2f] m/s",
                          obs->ele[0], obs->ele[1], obs->ele[sd], obs->ele[sd + 1]));
        
        linesOut.emplace_back(
            string_format("actions:       [% 1.3f,% 1.3f,% 1.3f,% 1.3f]", currentAction->ele[0], currentAction->ele[1],
                          currentAction->ele[2], currentAction->ele[3]));
        
        const double* distForce = forceDisturber->getLastForce();
        linesOut.emplace_back(
            string_format("disturbances:   [% 3.1f,% 3.1f,% 3.1f] N", distForce[0], distForce[1], distForce[2]));
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* eff_bpi = physicsManager->getBodyInfo("Effector");
            linesOut.emplace_back(string_format("effector mass:    % 1.3f kg", eff_bpi->body->m));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECMPblending> RegMPBlending("MPBlending");
    
}

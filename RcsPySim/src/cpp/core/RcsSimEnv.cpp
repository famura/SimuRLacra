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

#include "RcsSimEnv.h"
#include "action/ActionModel.h"
#include "action/ActionModelIK.h"
#include "observation/ObservationModel.h"
#include "initState/InitStateSetter.h"
#include "physics/ForceDisturber.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_VecNd.h>
#include <Rcs_basicMath.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>
#include <HUD.h>
#include <GraphNode.h>
#include <PhysicsNode.h>
#include <KeyCatcher.h>

#endif

#include <stdexcept>
#include <sstream>


Rcs::RcsSimEnv::RcsSimEnv(PropertySource* propertySource)
{
    // Set random seed
    Math_srand48(0);
    
    // Load experiment config
    config = ExperimentConfig::create(propertySource);
    
    // Check if all joints are position controlled for skipping a later inverse dynamics control (compliance)
    unsigned int numPosCtrlJoints = 0;
    RCSGRAPH_TRAVERSE_JOINTS(config->graph) {
            if (JNT->jacobiIndex != -1) {
                if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_POSITION) {
                    numPosCtrlJoints++;
                }
            }
        }
    allJointsPosCtrl = config->graph->nJ == numPosCtrlJoints;
    
    // Load physics parameter manager from config
    physicsManager = config->createPhysicsParameterManager();
    // Physics simulator instance is null at the start. It will only be created once reset is called.
    physicsSim = NULL;
    
    // Create force disturber
    disturber = config->createForceDisturber();
    
    // Load init state setter
    initStateSetter = config->createInitStateSetter();
    
    // Init temp matrices
    q_ctrl = MatNd_clone(config->graph->q);
    qd_ctrl = MatNd_clone(config->graph->q_dot);
    T_ctrl = MatNd_create(config->graph->dof, 1);
    
    // Other state-related stuff
    viewer = NULL;
    usePhysicsNode = propertySource->getPropertyBool("usePhysicsNode", false);
    hud = NULL;
    currentObservation = config->observationModel->getSpace()->createValueMatrix();
    currentAction = config->actionModel->getSpace()->createValueMatrix();
//    adWidgetHandle = -1;
    
    currentStep = 0;
    currentTime = 0;
//    lastStepReward = 0;
    
    // Init transition noise (unused if not explicitly flagged)
    transitionNoiseBuffer = NULL;
    tnbIndex = 0;
    transitionNoiseIncludeVelocity = propertySource->getPropertyBool("transitionNoiseIncludeVelocity");
}

Rcs::RcsSimEnv::~RcsSimEnv()
{
    // Close rendering, will free associated resources
    render("", true);
    
    MatNd_destroy(q_ctrl);
    MatNd_destroy(qd_ctrl);
    MatNd_destroy(T_ctrl);
    
    MatNd_destroy(currentAction);
    MatNd_destroy(currentObservation);
    
    delete initStateSetter;
    delete physicsSim;
    delete physicsManager;
    delete disturber;
    delete config;
}

MatNd* Rcs::RcsSimEnv::reset(PropertySource* domainParam, const MatNd* initState)
{
    // Set random seed
    Math_srand48(0);
    
    // Lock the graph
    std::unique_lock<std::mutex> lock(graphLock);
    
    // Set graph to initial state
    RcsGraph_setDefaultState(config->graph);
    RCSGRAPH_TRAVERSE_SENSORS(config->graph) {
        MatNd_setZero(SENSOR->rawData);
    }
    
    // Then, apply the initState
    if (initState && initStateSetter != NULL) {
        std::string errorMsg;
        if (!initStateSetter->getSpace()->contains(initState, &errorMsg)) {
            throw std::invalid_argument("init state space does not contain the init state: " + errorMsg);
        }
        initStateSetter->applyInitialState(initState);
    }
    
    // Rebuild physics sim with new parameters
    // We set it to NULL before creating the new one since there might be exceptions during parameter loading
    delete physicsSim;
    physicsSim = NULL;
    physicsSim = physicsManager->createSimulator(domainParam);
    
    // Reset control command vectors
    MatNd_copy(q_ctrl, config->graph->q);
    MatNd_setZero(qd_ctrl);
    MatNd_setZero(T_ctrl);
    
    // Reset stateful models
    config->actionModel->reset();
    config->observationModel->reset();
    // Reset the physics simulator
//    physicsSim->reset();  // this causes a GUI error
    physicsSim->resetTime();
    
    // Unlock the graph
    lock.unlock();
    
    // Reset local counters
    currentStep = 0;
    currentTime = 0;
//    lastStepReward = 0;
    
    // Return initial observation
    return config->observationModel->computeObservation(nullptr, config->dt);
}

MatNd* Rcs::RcsSimEnv::step(const MatNd* action, const MatNd* disturbance)
{
    REXEC(6) {
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Iteration " << currentStep << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
    }
    
    // Validate actions
    std::string iaMsg;
    if (!actionSpace()->checkDimension(action, &iaMsg)) {
        throw std::invalid_argument("Invalid action - " + iaMsg);
    }
    
    // Store the action for the HUD
    MatNd_copy(currentAction, action);
    
    // Lock the graph
    std::unique_lock<std::mutex> lock(graphLock);
    
    /*-------------------------------------------------------------------------*
     * Obtain simulator inputs from action model
     *-------------------------------------------------------------------------*/
    config->actionModel->computeCommand(q_ctrl, qd_ctrl, T_ctrl, action, config->dt);
    
    if (config->checkJointLimits) {
        std::ostringstream jvMsg;
        unsigned int numJV = 0;
        RCSGRAPH_TRAVERSE_JOINTS(config->graph) {
                // Skip the non-IK joints
                if (JNT->constrained) {
                    continue;
                }
                // Check the remaining joints' limits
                double qi = MatNd_get2(q_ctrl, JNT->jointIndex, 0);
                if ((qi < JNT->q_min) || (qi > JNT->q_max)) {
                    numJV++;
                    jvMsg << "  " << JNT->name << ": " << qi << " [" << JNT->q_min << ", " << JNT->q_max << "["
                          << std::endl;
                }
            }
        if (numJV > 0) {
            std::ostringstream os;
            os << "Detected " << numJV << " joint limit violations: " << std::endl << jvMsg.str();
            throw JointLimitException(os.str());
        }
    }
    
    if (disturbance != NULL && disturber != NULL) {
        // Check disturbance shape
        if (disturbance->m != 3 || disturbance->n != 1) {
            throw std::invalid_argument("Invalid disturbance");
        }
        disturber->apply(physicsSim, disturbance->ele);
    }
    
    /*-------------------------------------------------------------------------*
     * Inverse Dynamics in Joint Space (Compliance Control)
     *-------------------------------------------------------------------------*/
    if (!allJointsPosCtrl) {
        Rcs::ControllerBase::computeInvDynJointSpace(T_ctrl, config->graph, q_ctrl, 1000.);
    }
    
    /*-------------------------------------------------------------------------*
     * Call the Physics Simulation and Get the New Current State
     *-------------------------------------------------------------------------*/
    physicsSim->setControlInput(q_ctrl, qd_ctrl, T_ctrl);
    
    physicsSim->simulate(config->dt, config->graph);
    
    /*-------------------------------------------------------------------------*
     * Apply transition noise if desired
     *-------------------------------------------------------------------------*/
    if (transitionNoiseBuffer != NULL) {
        // Get noise set to use for this timestamp
        MatNd* noise;
        MatNd_fromStack(noise, getInternalStateDim(), 1);
        MatNd_getColumn(noise, tnbIndex, transitionNoiseBuffer);
        
        if (transitionNoiseIncludeVelocity) {
            // Apply first part to state and second part to velocity
            VecNd_addSelf(config->graph->q->ele, noise->ele, config->graph->dof);
            VecNd_addSelf(config->graph->q_dot->ele, &noise->ele[config->graph->dof], config->graph->dof);
        }
        else {
            // Apply to state values only
            MatNd_addSelf(config->graph->q, noise);
        }
        
        // Increment buffer index
        tnbIndex++;
        if (tnbIndex >= transitionNoiseBuffer->n) {
            tnbIndex = 0;
        }
        
        // cleanup
        MatNd_destroy(noise);
    }
    
    /*-------------------------------------------------------------------------*
     * Store new physics simulation state in the ExperimentConfig's graph
     *-------------------------------------------------------------------------*/
    // Apply forward kinematics
//    RcsGraph_setState(config->graph, NULL, config->graph->q_dot); // NULL leads to using the q vector from the graph
    RcsGraph_setState(config->graph, config->graph->q, config->graph->q_dot);
    
    // Unlock the graph
    lock.unlock();
    
    /*-------------------------------------------------------------------------*
     * For transition noise, we need to update the state of the physics
     * simulator to match the state of the graph after applying the noise.
     *-------------------------------------------------------------------------*/
    if (transitionNoiseBuffer != NULL) {
        RCSGRAPH_TRAVERSE_BODIES(config->graph) {
            physicsSim->applyTransform(BODY, BODY->A_BI);
            physicsSim->applyLinearVelocity(BODY, BODY->x_dot);
            physicsSim->applyAngularVelocity(BODY, BODY->omega);
        }
    }
    
    // Update step counters
    currentStep++;
    currentTime += config->dt;
    
    /*-------------------------------------------------------------------------*
     * Assemble results
     *-------------------------------------------------------------------------*/
    // compute observation
    MatNd* observation = config->observationModel->computeObservation(action, config->dt);
    MatNd_copy(currentObservation, observation);
    
    /*
    // compute state for goal monitor
    MatNd* state;
    if (config->stateModel == config->observationModel) {
        // use observation if it is the same
        state = observation;
    } else {
        // get explicitly if not
        state = config->stateModel->computeObservation(config->dt);
    }

    // compute reward
    auto stepReward = config->goalMonitor->stepReward(state, action);
    // cleanup temporary state mat if any
    if (state != observation) {
        MatNd_destroy(state);
    }

    lastStepReward = stepReward.reward;
    */
    
    return observation;
}

void Rcs::RcsSimEnv::render(std::string mode, bool close)
{
#ifdef GRAPHICS_AVAILABLE
    if (close) {
        // Close the viewer
        if (viewer) {
            hud = NULL;
            delete viewer;
            viewer = NULL;
        }
    }
    else if (mode == "human") {
        if (!viewer) {
            // Start viewer without shadows, since the shadow renderer has some issues when multiple simulations
            // are opened in the same process. You can still enable shadows via the key binding.
            viewer = new Rcs::Viewer(true, false);
            auto kc = new Rcs::KeyCatcher();
            
            // Make nodes resizable so it can update on physics param changes setting that for relevant nodes only
            // would be better, but it's not possible right now.
            // Visualize the IK controller graph (a.k.a. desired graph)
            auto amIK = config->actionModel->unwrap<ActionModelIK>();
            if (amIK != NULL) {
                Rcs::GraphNode* gDesNode = new Rcs::GraphNode(amIK->getController()->getGraph(), true, false);
                gDesNode->setGhostMode(true);
                viewer->add(gDesNode);
            }
            if (usePhysicsNode) {
                Rcs::PhysicsNode* pNode = new Rcs::PhysicsNode(physicsSim, true);
                viewer->add(pNode);
                pNode->physicsNd->togglePhysicsModel(); // switch off
                pNode->physicsNd->toggleGraphicsModel(); // switch on
            }
            else {
                Rcs::GraphNode* gNode = new Rcs::GraphNode(config->graph, true, false);
                gNode->toggleReferenceFrames();
                viewer->add(gNode);
            }
            
            hud = new Rcs::HUD();
            // HUD color from config
            std::string hudColor;
            if (config->properties->getProperty(hudColor, "hudColor") && !hudColor.empty()) {
                hud->setColor(hudColor.c_str());
            }
            hud->resize(1024, 100);
            
            viewer->setWindowSize(0, 0, 1024, 768);
            viewer->setUpdateFrequency(50.0);
            viewer->add(hud);
            viewer->add(kc);
            viewer->setBackgroundColor("PEWTER");
            
            // Experiment specific settings
            config->initViewer(viewer);
            
            viewer->runInThread(graphLock.native_handle());
            // viewer->toggleVideoRecording();  // start recording right from the start of the sim
        }
        // Update HUD
        auto hudText = config->getHUDText(
            currentTime, currentObservation, currentAction, physicsSim, physicsManager, disturber
        );
        hud->setText(hudText);
    }
#else
    // Warn about headless mode
    static bool warned = false;
    if (!warned)
    {
        warned = true;
        RMSGS("RcsPySim compiled in headless mode, thus rendering is not available");
    }
#endif
}

void Rcs::RcsSimEnv::toggleVideoRecording()
{
#ifdef GRAPHICS_AVAILABLE
    if (viewer == NULL) {
        // The viewer has not been opened yet, so do that.
        render("human", false);
        // If we start the recording right away, the recording target frame will be off.
        RPAUSE_MSG("Please move the graphics window so the window position is initialized properly."
                   "Press ENTER to continue.");
    }
    viewer->toggleVideoRecording();
#else
    // warn about headless mode
    static bool warned = false;
    if(!warned) {
        warned = true;
        RMSGS("RcsPySim compiled in headless mode; rendering not available");
    }
#endif
}

const Rcs::BoxSpace* Rcs::RcsSimEnv::observationSpace()
{
    return config->observationModel->getSpace();
}

const Rcs::BoxSpace* Rcs::RcsSimEnv::actionSpace()
{
    return config->actionModel->getSpace();
}

const Rcs::BoxSpace* Rcs::RcsSimEnv::initStateSpace()
{
    if (initStateSetter == NULL) {
        return NULL;
    }
    return initStateSetter->getSpace();
}

//const pybind11::dict Rcs::RcsSimEnv::domainParams()
//{
//    pybind11::dict domainParams = pybind11::dict("test"=1);
//    return domainParams;
//}

void Rcs::RcsSimEnv::setTransitionNoiseBuffer(const MatNd* tnb)
{
    if (tnb == NULL) {
        // remove it
        MatNd_destroy(transitionNoiseBuffer);
        transitionNoiseBuffer = NULL;
    }
    else if (tnb->m != getInternalStateDim()) {
        throw std::invalid_argument("Transition noise dimension must match internal state dimension");
    }
    else {
        // store copy
        MatNd_resizeCopy(&transitionNoiseBuffer, tnb);
        tnbIndex = 0;
    }
}

unsigned int Rcs::RcsSimEnv::getInternalStateDim()
{
    if (transitionNoiseIncludeVelocity) {
        return config->graph->dof*2;
    }
    return config->graph->dof;
}

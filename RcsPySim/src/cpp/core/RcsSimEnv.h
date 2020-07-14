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

#ifndef _RCSSIMENV_H_
#define _RCSSIMENV_H_

#include "ExperimentConfig.h"
#include "physics/PhysicsParameterManager.h"

#include "util/BoxSpace.h"
#include "util/nocopy.h"

#include <mutex>

namespace Rcs
{

class Viewer;

class HUD;

/**
 * Thrown if the joint limits are violated after applying the action.
 */
class JointLimitException : public std::runtime_error
{
    using runtime_error::runtime_error;
};

/**
 * Rcs-backed machine learning simulation environment.
 * This class provides a user-driven update loop. It is started by reset(), and then updated by step().
 */
class RcsSimEnv
{
public:
    /**
     * Create the environment from the given property source.
     *
     * @param propertySource configuration
     */
    explicit RcsSimEnv(PropertySource* propertySource);
    
    virtual ~RcsSimEnv();
    
    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(RcsSimEnv)
    
    /**
     * Reset the internal state in order to start a new rollout.
     *
     * @param domainParam physics params to use for this rollout
     * @param initState initial state for this rollout
     * @return observation of the initial state
     */
    virtual MatNd* reset(PropertySource* domainParam = PropertySource::empty(), const MatNd* initState = NULL);
    
    /**
     * Perform one environment step.
     *
     * @param action action vector
     * @param disturbance disturbance vector, e.g. a 3-dim force
     * @return observation after the step was processed
     * @throws JointLimitException if the joint limits were violated
     */
    virtual MatNd* step(const MatNd* action, const MatNd* disturbance = NULL);
    
    /**
     * Render the current state of the simulation.
     *
     * Should be called after each step call.
     * The RcsGraphics renderer runs mostly on it's own thread, so that isn't absolutely
     * necessairy, but it is required to update the HUD.
     *
     * @param mode only 'human' is supported at the moment, is also the default
     * @param close true to close the render window
     */
    virtual void render(std::string mode = "human", bool close = false);
    
    /** Start/stop video recording. */
    void toggleVideoRecording();
    
    /**
     * Observation space of this environment.
     * All valid observation values fit inside this.
     */
    const BoxSpace* observationSpace();
    
    /**
     * Action space of this environment.
     * All valid action values fit inside this.
     * The actions provided from Python are not projected to this space, i.e. this must be done on the Python side.
     */
    const BoxSpace* actionSpace();
    
    /**
     * Initial state space of this environment.
     * All valid initial state values fit inside this.
     */
    const BoxSpace* initStateSpace();
    
    /**
     * Set the transition noise buffer.
     * In order to avoid heavy stochastic computation in every step, the transition noise values are pregenerated.
     * The buffer should have a row count equal to getInternalStateDim(). Every column is a set of noise values
     * applied in one step. In the next step, the next column is used. If the last column is reached, the next step
     * will use the first column again.
     *
     * @param tnb transition noise buffer
     */
    void setTransitionNoiseBuffer(const MatNd* tnb);
    
    /** Internal state dimension, required for the transition noise buffer. */
    unsigned int getInternalStateDim();
    
    /** Configuration settings for the experiment */
    ExperimentConfig* getConfig()
    {
        return config;
    }
    
    /** Physics parameter management */
    PhysicsParameterManager* getPhysicsManager()
    {
        return physicsManager;
    }
    
    /** Observation from last step */
    MatNd* getCurrentObservation() const
    {
        return currentObservation;
    }
    
    /** Action from last step */
    MatNd* getCurrentAction() const
    {
        return currentAction;
    }

private:
    //! Guards for parallel access to graph (can happen from gui)
    std::mutex graphLock;
    
    //! Experiment configuration
    ExperimentConfig* config;
    bool allJointsPosCtrl;
    
    //! Physics simulator factory
    PhysicsParameterManager* physicsManager;
    //! Physics simulator
    PhysicsBase* physicsSim;
    //! Disturbance force simulator
    ForceDisturber* disturber;
    
    //! Initial state setter
    InitStateSetter* initStateSetter;
    
    //! Temporary matrices
    MatNd* q_ctrl;
    MatNd* qd_ctrl;
    MatNd* T_ctrl;
    
    //! Counters
    unsigned int currentStep;
    double currentTime;
    
    //! Transition noise values (every column is one set of noise values for every state variable)
    MatNd* transitionNoiseBuffer;
    unsigned int tnbIndex;
    bool transitionNoiseIncludeVelocity; // false if transition noise is only applied to state
    
    //! Observation and action at last step
    MatNd* currentObservation;
    MatNd* currentAction;
    
    //! Visualization
    Viewer* viewer;
    bool usePhysicsNode; // use PhysicsNode (can not be reset) or GraphicsNode for the viewer
    HUD* hud;
//    int adWidgetHandle;
};

} /* namespace Rcs */

#endif /* _RCSSIMENV_H_ */

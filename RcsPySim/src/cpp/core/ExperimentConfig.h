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

#ifndef _EXPERIMENTCONFIG_H_
#define _EXPERIMENTCONFIG_H_

#include "config/PropertySource.h"
#include "util/nocopy.h"

#include <Rcs_graph.h>

namespace Rcs
{

class ActionModel;

class ObservationModel;

class InitStateSetter;

class PhysicsParameterManager;

class PhysicsBase;

class ForceDisturber;

class HUD;

class Viewer;

/**
 * Create a collision model using the settings from the given RcsPySim config.
 *
 * @param graph graph to observe
 * @param config collision configuration
 * @return new collision model
 * @throw std::invalid_argument if the collision model could not be created.
 */
RcsCollisionMdl* RcsCollisionModel_createFromConfig(RcsGraph* graph, Rcs::PropertySource* config);

/**
 * Defines the experiment setup.
 * There should be one subclass per experiment type, IE BallOnPlate or QuanserQube.
 */
class ExperimentConfig
{
    // experiment catalog management
public:
    
    typedef ExperimentConfig* (* ExperimentConfigCreateFunction)();
    
    /**
     * Register a named experiment.
     * @param name experiment name
     * @param creator ExperimentConfig factory function
     */
    static void registerType(const char* name, ExperimentConfigCreateFunction creator);
    
    /**
     * Create the experiment configuration from the given property source.
     * This will load the "type" property, use that to select a ExperimentConfig implementation, and then call init()
     * to populate it.
     * @param properties property source to read configuration from. Takes ownership, even on error.
     * @return a new experiment config object.
     */
    static ExperimentConfig* create(PropertySource* properties);


public:
    ExperimentConfig();
    
    virtual ~ExperimentConfig();
    
    // not copy- or movable RCSPYSIM_NOCOPY_NOMOVE(ExperimentConfig)

protected:
    /**
     * Called to load data from the given properties.
     * You can override this to load additional data, but be sure to call the parent implementation.
     * @param properties property source to read configuration from. Takes ownership.
     */
    virtual void load(PropertySource* properties);
    
    /**
     * Create the action model. Read any configuration from this->properties.
     * @return the new action model
     */
    virtual ActionModel* createActionModel() = 0;
    
    /**
     * Create the observation model. Read any configuration from this->properties.
     * @return the new observation model
     */
    virtual ObservationModel* createObservationModel() = 0;

public:
    
    /**
     * Create the init state setter. Read any configuration from this->properties.
     * Since the init state setter is only needed for the simulation, it is not stored in the ExperimentConfig.
     * Instead, the simulation calls this method and manages the object on it's own.
     * The default implementation returns NULL to use the state from the graph file.
     * @return the new init state setter
     */
    virtual InitStateSetter* createInitStateSetter();
    
    /**
     * Create a model for artificial external disturbing forces.
     * The default implementation returns NULL to ignore this.
     * @return the new force disturber
     */
    virtual ForceDisturber* createForceDisturber();
    
    /**
     * Create the physics parameter manager. Read any configuration from this->properties.
     * Since the physics parameter manager is only needed for the simulation, it is not stored in the ExperimentConfig.
     * Instead, the simulation calls this method and manages the object on it's own.
     * This method calls populatePhysicsParameters to populate the parameter descriptors.
     * @return the new physics parameter manager
     */
    PhysicsParameterManager* createPhysicsParameterManager();
    
    /**
     * Called to update the HUD text for the viewer.
     * The default implementation will show the physics engine name, the current time and the last step reward.
     * @param[out] linesOut          vector of HUD lines. initially empty.
     * @param[in] currentTime        simulation time
     * @param[in] currentObservation latest observation
     * @param[in] currentAction      latest action
     * @param[in] simulator          physics simulator or NULL if none
     * @param[in] physicsManager     physics parameter manager or NULL if none
     * @param[in] forceDisturber     distruber which applies the forces to a given body
     * @return concatenated HUD lines
     */
    virtual void getHUDText(
        std::vector<std::string>& linesOut, double currentTime, const MatNd* currentObservation,
        const MatNd* currentAction, PhysicsBase* simulator,
        PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber);
    
    /**
     * Returns the concatenated HUD lines.
     * @param[in] currentTime        simulation time
     * @param[in] currentObservation latest observation
     * @param[in] currentAction      latest action
     * @param[in] simulator          physics simulator or NULL if none
     * @param[in] physicsManager     physics parameter manager or NULL if none
     * @param[in] forceDisturber     distruber which applies the forces to a given body
     * @return concatenated HUD lines
     */
    std::string getHUDText(
        double currentTime, const MatNd* currentObservation, const MatNd* currentAction,
        PhysicsBase* simulator, PhysicsParameterManager* physicsManager,
        ForceDisturber* forceDisturber);
    
    /**
     * Perform additional initialization on the viewer.
     * This could, for example, change the camera position or add additional visualization.
     * The default implementation does nothing.
     */
    virtual void initViewer(Rcs::Viewer* viewer);

protected:
    /**
     * Add the physics parameter descriptors to the given physics parameter manager.
     * @param manager parameter manager to populate.
     */
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager);

public:
    //! Property source (owned)
    PropertySource* properties;
    
    //! Graph description
    RcsGraph* graph;
    
    //! Action model (pluggable).
    ActionModel* actionModel;
    
    //! Observation model (pluggable) used to create the observation which will be returned from step()
    ObservationModel* observationModel;
    
    //! Collision model to use. Is based on the graph, so take care when using with IK etc.
    RcsCollisionMdl* collisionMdl;
    
    //! The time step [s]
    double dt;
    
    //! Flag to enable joint limit checking
    bool checkJointLimits;
};


/**
 * Create a static field of this type to register an experiment config type.
 */
template<class Config>
class ExperimentConfigRegistration
{
public:
    /**
     * Register the template type under the given name.
     * @param name experiment name
     */
    explicit ExperimentConfigRegistration(const char* name)
    {
        ExperimentConfig::registerType(name, ExperimentConfigRegistration::create);
    }
    
    /**
     * Creator function passed to ExperimentConfig::registerType.
     */
    static ExperimentConfig* create()
    {
        return new Config();
    }
    
};
    
} /* namespace Rcs */

#endif /* _EXPERIMENTCONFIG_H_ */

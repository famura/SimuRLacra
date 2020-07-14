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

#ifndef _PHYSICSPARAMETERMANAGER_H_
#define _PHYSICSPARAMETERMANAGER_H_

#include "BodyParamInfo.h"
#include "PhysicsParameterDescriptor.h"
#include "../util/nocopy.h"

#include <PhysicsBase.h>

#include <vector>
#include <list>

namespace Rcs
{

/**
 * Main physics parameter modification system.
 * Keeps a list of parameter descriptors, allows setting values on them
 * and transfers those values to the physics simulation.
 *
 */
class PhysicsParameterManager
{
public:
    /**
     * Constructor.
     * @param graph graph to modify
     * @param physicsEngineName name of physics engine to use
     * @param physicsConfigFile config file for the physics engine
     */
    PhysicsParameterManager(
        RcsGraph* graph,
        const std::string& physicsEngineName,
        const std::string& physicsConfigFile);
    
    virtual ~PhysicsParameterManager();
    
    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(PhysicsParameterManager)
    
    /**
     * Register a parameter descriptor operating on the given named body.
     * Takes ownership of the descriptor.
     * @param bodyName name of body to use
     * @param desc descriptor of the parameters to make changeable on the body.
     */
    void addParam(const char* bodyName, PhysicsParameterDescriptor* desc);
    
    /**
     * Get the BodyParamInfo object for the given named body, creating it if it doesn't exist.
     * @param bodyName name of body to look up.
     */
    BodyParamInfo* getBodyInfo(const char* bodyName);
    
    /**
     * Query current parameter values.
     */
    void getValues(PropertySink* outValues) const;
    
    /**
     * Create a new physics simulation using the given physics parameter values.
     * @param values parameter values to set
     * @return new physics simulator
     */
    PhysicsBase* createSimulator(PropertySource* values);

private:
    // graph to update
    RcsGraph* graph;
    
    // name of physics engine to use
    std::string physicsEngineName;
    
    // parsed physics engine configuration.
    PhysicsConfig* physicsConfig;
    
    // list of modifyable bodies. Uses std::list to get persistent references to elements.
    std::list<BodyParamInfo> bodyInfos;
    
    // list of parameter descriptors
    std::vector<PhysicsParameterDescriptor*> paramDescs;
};

} /* namespace Rcs */

#endif /* _PHYSICSPARAMETERMANAGER_H_ */

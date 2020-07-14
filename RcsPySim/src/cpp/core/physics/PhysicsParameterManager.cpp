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

#include "PhysicsParameterManager.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <PhysicsFactory.h>

namespace Rcs
{

PhysicsParameterManager::PhysicsParameterManager(
    RcsGraph* graph,
    const std::string& physicsEngineName,
    const std::string& physicsConfigFile) :
    graph(graph), physicsEngineName(physicsEngineName)
{
    physicsConfig = new PhysicsConfig(physicsConfigFile.c_str());
}

PhysicsParameterManager::~PhysicsParameterManager()
{
    for (auto pdesc : this->paramDescs) {
        delete pdesc;
    }
    delete physicsConfig;
}

void PhysicsParameterManager::addParam(const char* bodyName, PhysicsParameterDescriptor* desc)
{
    // obtain body info
    BodyParamInfo* bpi = getBodyInfo(bodyName);
    // init descriptor
    desc->init(bpi);
    // add it to list
    paramDescs.push_back(desc);
}

BodyParamInfo* PhysicsParameterManager::getBodyInfo(const char* bodyName)
{
    // check if used already
    for (BodyParamInfo& existing : bodyInfos) {
        if (STREQ(existing.body->name, bodyName)) {
            return &existing;
        }
    }
    // not found, so add
    bodyInfos.emplace_back(graph, bodyName, physicsConfig);
    return &bodyInfos.back();
}


void PhysicsParameterManager::getValues(PropertySink* outValues) const
{
    for (auto pdesc : this->paramDescs) {
        pdesc->getValues(outValues);
    }
}

PhysicsBase* PhysicsParameterManager::createSimulator(PropertySource* values)
{
    for (auto pdesc : this->paramDescs) {
        pdesc->setValues(values);
    }
    
    return PhysicsFactory::create(physicsEngineName.c_str(), graph, physicsConfig);
}

} /* namespace Rcs */

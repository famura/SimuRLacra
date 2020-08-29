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

#include "ControlPolicy.h"
#include "../config/PropertySource.h"

#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>

#include <map>
#include <sstream>

namespace Rcs
{

// The policy type registry
static std::map<std::string, ControlPolicy::ControlPolicyCreateFunction> registry;

void ControlPolicy::registerType(
    const char* name,
    ControlPolicy::ControlPolicyCreateFunction creator)
{
    // Store in registry
    registry[name] = creator;
}

ControlPolicy* ControlPolicy::create(const char* name, const char* dataFile)
{
    // Lookup factory for type
    auto iter = registry.find(name);
    if (iter == registry.end()) {
        std::ostringstream os;
        os << "Unknown control policy type '" << name << "'!";
        throw std::invalid_argument(os.str());
    }
    
    // Find data file
    char filepath[256];
    bool found = Rcs_getAbsoluteFileName(dataFile, filepath);
    if (!found) {
        // file does not exist
        Rcs_printResourcePath();
        std::ostringstream os;
        os << "Policy file '" << dataFile << "' does not exist!";
        throw std::invalid_argument(os.str());
    }
    
    // Create instance
    return iter->second(filepath);
}

ControlPolicy* ControlPolicy::create(PropertySource* config)
{
    std::string policyType;
    std::string policyFile;
    RCHECK(config->getProperty(policyType, "type"));
    RCHECK(config->getProperty(policyFile, "file"));
    return Rcs::ControlPolicy::create(policyType.c_str(), policyFile.c_str());
}

std::vector<std::string> ControlPolicy::getTypeNames()
{
    std::vector<std::string> names;
    for (auto& elem : registry) {
        names.push_back(elem.first);
    }
    return names;
}

ControlPolicy::ControlPolicy()
{
    // Does nothing
}

ControlPolicy::~ControlPolicy()
{
    // Does nothing
}

void ControlPolicy::reset()
{
    // Does nothing by default
}

} /* namespace Rcs */

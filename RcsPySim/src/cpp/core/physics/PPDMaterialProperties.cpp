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

#include "PPDMaterialProperties.h"

#include <libxml/tree.h>
#include <array>

namespace Rcs
{

// Vortex extended property list
static const char* extended_xml_material_props[] = {
//    "slip", // TODO detect which solver bullet is suing and then automatically add this entry
    "compliance",
};

PPDMaterialProperties::PPDMaterialProperties() {}

PPDMaterialProperties::~PPDMaterialProperties() = default;


void PPDMaterialProperties::getValues(PropertySink* outValues)
{
    std::string prefixedName;
    // Bullet and Vortex
    prefixedName = bodyParamInfo->paramNamePrefix + "friction_coefficient";
    outValues->setProperty(prefixedName.c_str(), bodyParamInfo->material.getFrictionCoefficient());
    prefixedName = bodyParamInfo->paramNamePrefix + "rolling_friction_coefficient";
    outValues->setProperty(prefixedName.c_str(), bodyParamInfo->material.getRollingFrictionCoefficient());
    prefixedName = bodyParamInfo->paramNamePrefix + "restitution";
    outValues->setProperty(prefixedName.c_str(), bodyParamInfo->material.getRestitution());
    prefixedName = bodyParamInfo->paramNamePrefix + "slip";
    double slip = 0; // TODO @Michael: there is no bodyParamInfo->material.getSlip()
    bodyParamInfo->material.getDouble("slip", slip);
    outValues->setProperty(prefixedName.c_str(), slip);
    
    // Extension properties stored in the config xml-file
    for (auto propname : extended_xml_material_props) {
        auto configValue = xmlGetProp(bodyParamInfo->material.materialNode, BAD_CAST propname);
        if (configValue != nullptr) {
            prefixedName = bodyParamInfo->paramNamePrefix + propname;
            outValues->setProperty(prefixedName.c_str(), configValue);  // types aren't well defined, assuming string
        }
    }
}

void PPDMaterialProperties::setValues(PropertySource* inValues)
{
    std::string prefixedName;
    double value;
    // Bullet and Vortex
    prefixedName = bodyParamInfo->paramNamePrefix + "friction_coefficient";
    if (inValues->getProperty(value, prefixedName.c_str())) {
        bodyParamInfo->material.setFrictionCoefficient(value);
    }
    prefixedName = bodyParamInfo->paramNamePrefix + "rolling_friction_coefficient";
    if (inValues->getProperty(value, prefixedName.c_str())) {
        bodyParamInfo->material.setRollingFrictionCoefficient(value);
    }
    prefixedName = bodyParamInfo->paramNamePrefix + "restitution";
    if (inValues->getProperty(value, prefixedName.c_str())) {
        bodyParamInfo->material.setRestitution(value);
    }
    prefixedName = bodyParamInfo->paramNamePrefix + "slip";
    if (inValues->getProperty(value, prefixedName.c_str())) {
//        bodyParamInfo->material.setSlip(value); / TODO @Michael: there is no bodyParamInfo->material.setSlip()
        bodyParamInfo->material.setDouble("slip", value);
    }
    
    // Extension properties stored in the config xml-file
    std::string configValue;
    for (auto propname : extended_xml_material_props) {
        prefixedName = bodyParamInfo->paramNamePrefix + propname;
        // Transfer config value as string to support all possible kinds
        if (inValues->getProperty(configValue, prefixedName.c_str())) {
            // Store in the xml-file
            xmlSetProp(bodyParamInfo->material.materialNode, BAD_CAST propname, BAD_CAST configValue.c_str());
        }
    }
}

} /* namespace Rcs */

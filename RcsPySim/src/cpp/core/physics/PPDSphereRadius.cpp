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

#include "PPDSphereRadius.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

PPDSphereRadius::PPDSphereRadius(std::string prevBodyName, unsigned int shapeIdx, unsigned int shapeIdxPrevBody) :
    PPDSingleVar("radius", BodyParamInfo::MOD_SHAPE,
                 [](BodyParamInfo& bpi) -> double& { return bpi.body->shape[0]->extents[0]; }),
    prevBodyName(std::move(prevBodyName)), shapeIdx(shapeIdx), shapeIdxPrevBody(shapeIdxPrevBody) {}

PPDSphereRadius::~PPDSphereRadius() = default;

void PPDSphereRadius::setValues(PropertySource* inValues)
{
    // Adapt properties
    PPDSingleVar::setValues(inValues);
    
    // Check if the ball position variable is relative to another body
    RcsBody* prevBody = RcsGraph_getBodyByName(bodyParamInfo->graph, prevBodyName.c_str());
    double zOffset = 0.;
    if (prevBody != NULL) {
        if (bodyParamInfo->body->parent == prevBody) {
            // Sphere rigid body coordinates are relative
        }
        else {
            // The sphere's rigid body coordinates are absolute, shift them accordingly
            // Note: this assumes that the refBody is level, i.e. not tilted
            zOffset = prevBody->A_BI->org[2];
        }
        
        if (prevBody->shape[shapeIdxPrevBody]->type == RCSSHAPE_TYPE::RCSSHAPE_BOX) {
            zOffset += prevBody->shape[shapeIdxPrevBody]->extents[2]/2.;
        }
        else if (prevBody->shape[shapeIdxPrevBody]->type == RCSSHAPE_TYPE::RCSSHAPE_CYLINDER) {
            zOffset += prevBody->shape[shapeIdxPrevBody]->extents[2]/2.;
        }
        else if (prevBody->shape[shapeIdxPrevBody]->type == RCSSHAPE_TYPE::RCSSHAPE_SPHERE) {
            zOffset += prevBody->shape[shapeIdxPrevBody]->extents[0];
        }
        else {
            REXEC(4) {
                std::cout << "No default vertical offset found for previous body shape " <<
                          prevBody->shape[0]->type << std::endl;
            }
        }
    }
    else {
        REXEC(4) {
            std::cout << "No reference body found for adding an offset to the randomized sphere's position! " <<
                      "Received " << prevBodyName << std::endl;
        }
    }
    
    // Adapt the sphere's z-position
    double newRadius = bodyParamInfo->body->shape[shapeIdx]->extents[0];
    bodyParamInfo->graph->q->ele[bodyParamInfo->body->jnt->jointIndex + 2] = zOffset + newRadius;
    
    // Make sure the state is propagated
    RcsGraph_setState(bodyParamInfo->graph, NULL, bodyParamInfo->graph->q_dot);
    
    RLOG(4, "New radius = %f; New z-position = %f", newRadius, zOffset + newRadius);
}

void PPDSphereRadius::init(BodyParamInfo* bodyParamInfo)
{
    // Check if the ball is valid
    PPDSingleVar::init(bodyParamInfo);
    RCHECK_MSG(bodyParamInfo->body->shape != NULL, "Invalid ball body %s", bodyParamInfo->body->name);
    RCHECK_MSG(bodyParamInfo->body->shape[shapeIdx] != NULL, "Invalid ball body %s", bodyParamInfo->body->name);
    RCHECK_MSG(bodyParamInfo->body->shape[shapeIdx]->type == RCSSHAPE_SPHERE, "Invalid ball body %s",
               bodyParamInfo->body->name);
    RCHECK_MSG((bodyParamInfo->body->shape[shapeIdx]->computeType & RCSSHAPE_COMPUTE_PHYSICS) != 0,
               "Invalid ball body %s", bodyParamInfo->body->name);
    RCHECK_MSG(bodyParamInfo->body->rigid_body_joints, "Invalid ball body %s", bodyParamInfo->body->name);
}

} /* namespace Rcs */

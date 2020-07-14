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

#include "PPDRodLength.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_body.h>
#include <Rcs_Vec3d.h>

Rcs::PPDRodLength::PPDRodLength() : rodShape(NULL), childJoint(NULL)
{
    // Initial rod direction is along the z-axis
    Vec3d_setUnitVector(rodDirection, 2);
    Vec3d_setZero(rodOffset);
    Vec3d_setZero(jointOffset);
}

void Rcs::PPDRodLength::init(Rcs::BodyParamInfo* bpi)
{
    PhysicsParameterDescriptor::init(bpi);
    
    propertyName = bpi->paramNamePrefix + "length";
    
    // Locate the rod shape
    RCSBODY_TRAVERSE_SHAPES(bpi->body) {
        if (SHAPE->type == RCSSHAPE_CYLINDER && (SHAPE->computeType & RCSSHAPE_COMPUTE_PHYSICS) != 0) {
            rodShape = SHAPE;
            break;
        }
    }
    RCHECK_MSG(rodShape, "No cylinder shape found on body %s.", bpi->body->name);
    
    // Get rod length as set in xml
    double initialRodLength = rodShape->extents[2];
    
    // The rod direction in shape frame is along the z-axis
    Vec3d_setUnitVector(rodDirection, 2);
    
    // Convert to body frame
    Vec3d_transRotateSelf(rodDirection, rodShape->A_CB.rot);
    
    // The rod has, per se, no real good way to determine if it's in the positive or negative direction.
    // however, we can guess by looking on which side of the body origin the rod center is
    double curDistToOrigin = Vec3d_innerProduct(rodDirection, rodShape->A_CB.org);
    if (curDistToOrigin < 0) {
        // the rod goes into -direction, so invert direction
        Vec3d_constMulSelf(rodDirection, -1);
    }
    
    // The rod offset is the difference between the rod's initial start and the body's origin
    // shapePos = offset + rodDir * length / 2
    // => offset = shapePos - rodDir * length / 2
    Vec3d_constMulAndAdd(rodOffset, rodShape->A_CB.org, rodDirection, -initialRodLength/2);
    
    if (STREQ(bpi->body->name, "Arm")) {
        // locate pendulum body as child if any
        RcsBody* pendulumChild = NULL;
        for (RcsBody* child = bpi->body->firstChild; child != NULL; child = child->next) {
            if (STREQ(child->name, "Pendulum")) {
                pendulumChild = child;
                break;
            }
        }
        RCHECK_MSG(pendulumChild, "Arm body doesn't have a pendulum child.");
        
        // Extract joint
        childJoint = pendulumChild->jnt;
        
        // Compute offset between rod end and joint if any
        Vec3d_constMulAndAdd(jointOffset, childJoint->A_JP->org, rodDirection, -initialRodLength);
    }
}

void Rcs::PPDRodLength::setValues(Rcs::PropertySource* inValues)
{
    double length;
    if (!inValues->getProperty(length, propertyName.c_str())) {
        // not set, ignore
        return;
    }
    
    // Set shape extends
    rodShape->extents[2] = length;
    // The shape origin is in the middle of the rod, the body origin at the end. Thus, shift the shape.
    Vec3d_constMulAndAdd(rodShape->A_CB.org, rodOffset, rodDirection, length/2);
    
    // Also adjust child joint if needed
    if (childJoint != NULL) {
        // the joint should be at the end of the rod.
        Vec3d_constMulAndAdd(childJoint->A_JP->org, jointOffset, rodDirection, length);
    }
}

void Rcs::PPDRodLength::getValues(PropertySink* outValues)
{
    outValues->setProperty(propertyName.c_str(), rodShape->extents[2]);
}

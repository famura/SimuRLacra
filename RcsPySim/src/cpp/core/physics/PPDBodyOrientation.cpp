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

#include "PPDBodyOrientation.h"
#include "PPDSingleVar.h"

#include <Rcs_typedef.h>
#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>

#include <stdexcept>

namespace Rcs
{

void Vec33d_setZero(double v[3][3])
{
    for (unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            v[i][j] = 0.0;
        }
    }
}

void Vec33d_add(double dst[3][3], const double v1[3][3], const double v2[3][3])
{
    for (unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            dst[i][j] = v1[i][j] + v2[i][j];
        }
    }
}

PPDBodyOrientation::PPDBodyOrientation(const bool includeA, const bool includeB, const bool includeC)
{
    Vec33d_setZero(initRot); // undefined
    Vec3d_setZero(offsetRot);
    
    if (includeA) {
        addChild(new PPDSingleVar<double>(
            "rot_offset_a", BodyParamInfo::MOD_ORIENTATION,
            [this](BodyParamInfo& bpi) -> double& { return (offsetRot[0]); })
        );
    }
    if (includeB) {
        addChild(new PPDSingleVar<double>(
            "rot_offset_b", BodyParamInfo::MOD_ORIENTATION,
            [this](BodyParamInfo& bpi) -> double& { return (offsetRot[1]); })
        );
    }
    if (includeC) {
        addChild(new PPDSingleVar<double>(
            "rot_offset_c", BodyParamInfo::MOD_ORIENTATION,
            [this](BodyParamInfo& bpi) -> double& { return (offsetRot[2]); })
        );
    }
    
    if (getChildren().empty()) {
        throw std::invalid_argument("No position specified for PPDBodyOrientation!");
    }
}

PPDBodyOrientation::~PPDBodyOrientation() = default;

void PPDBodyOrientation::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);
    
    // Copy the rotation about the world x, y, and z axis. Here, a special case of MatNd_copy().
    memmove(initRot, bpi->body->A_BP->rot, 3*3*sizeof(double));
}

void PPDBodyOrientation::setValues(PropertySource* inValues)
{
    PPDCompound::setValues(inValues);
    
    // Copy the rotation about the world x, y, and z axis. Here, a special case of MatNd_copy().
    double tmpRot[3][3];
    memmove(tmpRot, initRot, 3*3*sizeof(double));
    
    // Apply the orientation to the body. The default value for offsetRot is zero.
    Mat3d_rotateSelfAboutXYZAxis(tmpRot, 0, offsetRot[0]);
    Mat3d_rotateSelfAboutXYZAxis(tmpRot, 1, offsetRot[1]);
    Mat3d_rotateSelfAboutXYZAxis(tmpRot, 2, offsetRot[2]);
    
    // Copy the rotation about the world x, y, and z axis. Here, a special case of MatNd_copy().
    memmove(this->bodyParamInfo->body->A_BP->rot, tmpRot, 3*3*sizeof(double));
}

} /* namespace Rcs */

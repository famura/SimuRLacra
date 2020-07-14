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

#include "PPDBodyPosition.h"
#include "PPDSingleVar.h"

#include <Rcs_typedef.h>
#include <Rcs_body.h>
#include <Rcs_shape.h>
#include <Rcs_math.h>
#include <Rcs_Vec3d.h>

#include <stdexcept>

namespace Rcs
{


PPDBodyPosition::PPDBodyPosition(const bool includeX, const bool includeY, const bool includeZ)
{
    Vec3d_setZero(initPos);
    Vec3d_setZero(offset);
    
    if (includeX) {
        addChild(new PPDSingleVar<double>(
            "pos_offset_x", BodyParamInfo::MOD_POSITION, [this](BodyParamInfo& bpi) -> double& { return (offset[0]); })
        );
    }
    if (includeY) {
        addChild(new PPDSingleVar<double>(
            "pos_offset_y", BodyParamInfo::MOD_POSITION, [this](BodyParamInfo& bpi) -> double& { return (offset[1]); })
        );
    }
    if (includeZ) {
        addChild(new PPDSingleVar<double>(
            "pos_offset_z", BodyParamInfo::MOD_POSITION, [this](BodyParamInfo& bpi) -> double& { return (offset[2]); })
        );
    }
    
    if (getChildren().empty()) {
        throw std::invalid_argument("No position specified for PPDBodyPosition!");
    }
}

PPDBodyPosition::~PPDBodyPosition() = default;

void PPDBodyPosition::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);
    
    Vec3d_copy(initPos, bpi->body->A_BP->org);
}

void PPDBodyPosition::setValues(PropertySource* inValues)
{
    PPDCompound::setValues(inValues);
    
    // Apply the position offset to the body
    Vec3d_add(this->bodyParamInfo->body->A_BP->org, initPos, offset);
}

} /* namespace Rcs */

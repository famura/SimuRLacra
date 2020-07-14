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

#include "PPDBoxExtents.h"
#include "PPDSingleVar.h"

#include <Rcs_typedef.h>

#include <stdexcept>

namespace Rcs
{

#define DEF_EXTENT_PARAM(name, modflag, var) \
    addChild(new PPDSingleVar<double>((name), (modflag), [this](BodyParamInfo& bpi) -> double& {return (var);}))

PPDBoxExtents::PPDBoxExtents(
    unsigned int shapeIdx,
    const bool includeLength,
    const bool includeWidth,
    const bool includeHeight) : shapeIdx(shapeIdx)
{
    // Add the children of type PPDSingleVar
    if (includeLength) {
        DEF_EXTENT_PARAM("length", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[0]);
    }
    if (includeWidth) {
        DEF_EXTENT_PARAM("width", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[1]);
    }
    if (includeHeight) {
        DEF_EXTENT_PARAM("height", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[2]);
    }
    
    if (getChildren().empty()) {
        throw std::invalid_argument("No position specified for PPDBoxExtents!");
    }
}

PPDBoxExtents::~PPDBoxExtents() = default;

void PPDBoxExtents::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);
    
    if (bpi->body->shape[this->shapeIdx]->type != RCSSHAPE_TYPE::RCSSHAPE_BOX) {
        throw std::invalid_argument("Using the PPDBoxExtents on a non-box shape!");
    }
}

void PPDBoxExtents::setValues(PropertySource* inValues)
{
    // Change the shape via the childrens' setValues() method without any other adaption
    PPDCompound::setValues(inValues);
}


PPDCubeExtents::PPDCubeExtents(unsigned int shapeIdx) : shapeIdx(shapeIdx)
{
    // Add the children of type PPDSingleVar
    DEF_EXTENT_PARAM("size", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[0]);
    DEF_EXTENT_PARAM("size", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[1]);
    DEF_EXTENT_PARAM("size", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[2]);
    
    if (getChildren().empty()) {
        throw std::invalid_argument("No position specified for PPDBoxExtents!");
    }
}

PPDCubeExtents::~PPDCubeExtents() = default;

void PPDCubeExtents::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);
    
    if (bpi->body->shape[this->shapeIdx]->type != RCSSHAPE_TYPE::RCSSHAPE_BOX) {
        throw std::invalid_argument("Using the PPDCubeExtents on a non-box shape!");
    }
}

void PPDCubeExtents::setValues(PropertySource* inValues)
{
    // Change the shape via the childrens' setValues() method without any other adaption
    PPDCompound::setValues(inValues);
}


} /* namespace Rcs */

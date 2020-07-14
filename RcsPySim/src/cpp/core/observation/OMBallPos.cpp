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

#include "OMBallPos.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>

namespace Rcs
{

OMBallPos::OMBallPos(RcsGraph* graph) :
    OMBodyStateLinear(graph, "Ball", "Plate"), ballRadius(0)
{
    // reset to update ball radius
    reset();
    
    // find plate dimensions
    RcsShape* plateShape = NULL;
    RCSBODY_TRAVERSE_SHAPES(getTask()->getRefBody()) {
        if (SHAPE->type == RCSSHAPE_BOX) {
            // found the shape
            plateShape = SHAPE;
            break;
        }
    }
    RCHECK_MSG(plateShape, "Plate body must have a box shape.");
    double plateWidth = plateShape->extents[0];
    double plateHeight = plateShape->extents[1];
    
    // use plate dims to initialize limits, z limits are arbitrary
    setMinState({-plateWidth/2, -plateHeight/2, -ballRadius - 0.1});
    setMaxState({+plateWidth/2, +plateHeight/2, +ballRadius + 0.1});
    // velocity limit is arbitrary too.
    setMaxVelocity(5.0);
    
}

OMBallPos::~OMBallPos()
{
    // nothing to destroy specifically
}

void OMBallPos::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    OMTask::computeObservation(state, velocity, currentAction, dt);
    // remove ball radius from z pos
    state[2] -= ballRadius;
}

void OMBallPos::reset()
{
    // update ball radius in case it changed
    RCSBODY_TRAVERSE_SHAPES(getTask()->getEffector()) {
        if (SHAPE->type == RCSSHAPE_SPHERE) {
            // found the ball shape
            ballRadius = SHAPE->extents[0];
            break;
        }
    }
}

} /* namespace Rcs */

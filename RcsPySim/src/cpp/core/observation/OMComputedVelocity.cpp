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

#include "OMComputedVelocity.h"

#include <Rcs_VecNd.h>

namespace Rcs
{

OMComputedVelocity::OMComputedVelocity() : lastState(NULL) {}

OMComputedVelocity::~OMComputedVelocity()
{
    // destroy last state mat
    MatNd_destroy(lastState);
}

void
OMComputedVelocity::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    // compute current state
    computeState(state, currentAction, dt);
    
    // compute velocity as forward derivative
    VecNd_sub(velocity, state, lastState->ele, getStateDim());
    VecNd_constMulSelf(velocity, 1/dt, getStateDim());
    
    // save state for next step
    VecNd_copy(lastState->ele, state, getStateDim());
}

void OMComputedVelocity::reset()
{
    // create last state storage if needed
    if (lastState == NULL) {
        lastState = MatNd_create(getStateDim(), 1);
    }
    // initialize last state storage.
    computeState(lastState->ele, NULL, 0.0);
}

} /* namespace Rcs */

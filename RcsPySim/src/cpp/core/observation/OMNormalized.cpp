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

#include "OMNormalized.h"

#include <stdexcept>
#include <sstream>
#include <cmath>

static void
validateAndOverride(MatNd* bound, Rcs::PropertySource* override, const char* boundName, const Rcs::BoxSpace* space)
{
    // Check each element
    auto& names = space->getNames();
    unsigned int nEle = bound->size;
    for (unsigned int i = 0; i < nEle; ++i) {
        auto bn = names[i];
        // Try to load override
        override->getProperty(bound->ele[i], bn.c_str());
        
        // Validate element is bounded now.
        if (std::isinf(bound->ele[i])) {
            std::ostringstream os;
            os << bn << " entry of " << boundName
               << " bound is infinite and not overridden. Cannot apply normalization.";
            throw std::invalid_argument(os.str());
        }
    }
}

Rcs::OMNormalized::OMNormalized(
    Rcs::ObservationModel* wrapped, PropertySource* overrideMin, PropertySource* overrideMax) : wrapped(wrapped)
{
    // Get inner model bounds with optional overrides
    MatNd* iModMin = NULL;
    MatNd* iModMax = NULL;
    MatNd_clone2(iModMin, wrapped->getSpace()->getMin())
    MatNd_clone2(iModMax, wrapped->getSpace()->getMax())
    
    validateAndOverride(iModMin, overrideMin, "lower", wrapped->getSpace());
    validateAndOverride(iModMax, overrideMax, "upper", wrapped->getSpace());
    
    // Compute scale and shift from inner model bounds
    // Shift is selected so that the median of min and max is 0
    // shift = min + (max - min)/2
    shift = MatNd_clone(iModMax);
    MatNd_subSelf(shift, iModMin);
    MatNd_constMulSelf(shift, 0.5);
    MatNd_addSelf(shift, iModMin);
    
    // scale = (max - min)/2
    scale = MatNd_clone(iModMax);
    MatNd_subSelf(scale, iModMin);
    MatNd_constMulSelf(scale, 0.5);
    
    // Cleanup temporary matrices
    MatNd_destroy(iModMax);
    MatNd_destroy(iModMin);
}

Rcs::OMNormalized::~OMNormalized()
{
    // Free matrices
    MatNd_destroy(shift);
    MatNd_destroy(scale);
    
    delete wrapped;
}

unsigned int Rcs::OMNormalized::getStateDim() const
{
    return wrapped->getStateDim();
}

unsigned int Rcs::OMNormalized::getVelocityDim() const
{
    return wrapped->getVelocityDim();
}

void Rcs::OMNormalized::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    // Query inner model
    wrapped->computeObservation(state, velocity, currentAction, dt);
    
    // Normalize values
    unsigned int sdim = getStateDim();
    for (unsigned int i = 0; i < sdim; ++i) {
        state[i] = (state[i] - shift->ele[i])/scale->ele[i];
    }
    for (unsigned int i = 0; i < getVelocityDim(); ++i) {
        velocity[i] = (velocity[i] - shift->ele[i + sdim])/scale->ele[i + sdim];
    }
}

void Rcs::OMNormalized::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    // query inner model
    wrapped->getLimits(minState, maxState, maxVelocity);
    // report actual scaled bounds, not explicit overrides
    unsigned int sdim = getStateDim();
    for (unsigned int i = 0; i < sdim; ++i) {
        minState[i] = (minState[i] - shift->ele[i])/scale->ele[i];
        maxState[i] = (maxState[i] - shift->ele[i])/scale->ele[i];
    }
    for (unsigned int i = 0; i < getVelocityDim(); ++i) {
        maxVelocity[i] = (maxVelocity[i] - shift->ele[i + sdim])/scale->ele[i + sdim];
    }
}

void Rcs::OMNormalized::reset()
{
    wrapped->reset();
}

std::vector<std::string> Rcs::OMNormalized::getStateNames() const
{
    return wrapped->getStateNames();
}

std::vector<std::string> Rcs::OMNormalized::getVelocityNames() const
{
    return wrapped->getVelocityNames();
}

std::vector<Rcs::ObservationModel*> Rcs::OMNormalized::getNested() const
{
    return {wrapped};
}

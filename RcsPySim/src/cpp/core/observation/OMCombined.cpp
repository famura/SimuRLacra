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

#include "OMCombined.h"

namespace Rcs
{

OMCombined::~OMCombined()
{
    for (auto part : parts) {
        delete part;
    }
}

void OMCombined::addPart(ObservationModel* part)
{
    parts.push_back(part);
}

unsigned int Rcs::OMCombined::getStateDim() const
{
    // do it for all parts
    unsigned int sumdim = 0;
    for (auto part : parts) {
        sumdim += part->getStateDim();
    }
    return sumdim;
}

unsigned int OMCombined::getVelocityDim() const
{
    // do it for all parts
    unsigned int sumdim = 0;
    for (auto part : parts) {
        sumdim += part->getVelocityDim();
    }
    return sumdim;
}

void Rcs::OMCombined::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    // do it for all parts
    for (auto part : parts) {
        part->computeObservation(state, velocity, currentAction, dt);
        state += part->getStateDim();
        velocity += part->getVelocityDim();
    }
}

void Rcs::OMCombined::getLimits(
    double* minState, double* maxState,
    double* maxVelocity) const
{
    // do it for all parts
    for (auto part : parts) {
        part->getLimits(minState, maxState, maxVelocity);
        minState += part->getStateDim();
        maxState += part->getStateDim();
        maxVelocity += part->getVelocityDim();
    }
}

std::vector<std::string> Rcs::OMCombined::getStateNames() const
{
    // reserve dim out vars
    std::vector<std::string> out;
    out.reserve(getStateDim());
    // concatenate names from parts
    for (auto part : parts) {
        auto pnames = part->getStateNames();
        // move the elements from pnames since it is a copy anyways.
        std::move(pnames.begin(), pnames.end(), std::inserter(out, out.end()));
    }
    return out;
}

std::vector<std::string> OMCombined::getVelocityNames() const
{
    // reserve dim out vars
    std::vector<std::string> out;
    out.reserve(getVelocityDim());
    // concatenate names from parts
    for (auto part : parts) {
        auto pnames = part->getVelocityNames();
        // move the elements from pnames since it is a copy anyways.
        std::move(pnames.begin(), pnames.end(), std::inserter(out, out.end()));
    }
    return out;
}

void OMCombined::reset()
{
    // do it for all parts
    for (auto part : parts) {
        part->reset();
    }
}

std::vector<ObservationModel*> OMCombined::getNested() const
{
    return parts;
}

} /* namespace Rcs */

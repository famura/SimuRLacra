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

#include "OMPartial.h"

#include <Rcs_macros.h>

#include <algorithm>
#include <sstream>

namespace Rcs
{

using IndexList = OMPartial::IndexList;

// Helpers for the constructor
static IndexList loadIndexList(IndexList input, unsigned int dim, bool exclude, const char* category)
{
    // Verify
    for (auto idx : input) {
        if (idx >= dim) {
            std::ostringstream os;
            os << (exclude ? "Excluded " : "Selected ") << category << " index " << idx
               << " is outside of the value dimension "
               << dim;
            throw std::invalid_argument(os.str());
        }
    }
    
    // Invert if needed
    if (exclude) {
        IndexList out;
        for (unsigned int idx = 0; idx < dim; ++idx) {
            // add if not in index list
            if (std::find(input.begin(), input.end(), idx) == input.end()) {
                out.push_back(idx);
            }
        }
        return out;
    }
    else {
        return input;
    }
}

static IndexList loadMask(const std::vector<bool>& mask, unsigned int dim, bool exclude, const char* category)
{
    // Verify
    if (mask.size() != dim) {
        std::ostringstream os;
        os << category << " mask size " << mask.size() << " does not match value dimension " << dim;
        throw std::invalid_argument(os.str());
    }
    // Convert to index list
    IndexList out;
    for (unsigned int idx = 0; idx < dim; ++idx) {
        // add true entries if exclude is false, or false entries if exclude is true
        if (mask[idx] == !exclude) {
            out.push_back(idx);
        }
    }
    return out;
}

OMPartial::OMPartial(
    ObservationModel* wrapped,
    IndexList indices, bool exclude) :
    wrapped(wrapped),
    keptStateIndices(loadIndexList(indices, wrapped->getStateDim(), exclude, "state"))
{
    if (wrapped->getVelocityDim() == wrapped->getStateDim()) {
        // Use state for velocity
        keptVelocityIndices = keptStateIndices;
    }
    else if (wrapped->getVelocityDim() != 0) {
        // Use explicit ctor
        throw std::invalid_argument("Cannot use same selection for state and velocity since their sizes don't match.");
    }
}


OMPartial::OMPartial(ObservationModel* wrapped, IndexList stateIndices, IndexList velocityIndices, bool exclude) :
    wrapped(wrapped),
    keptStateIndices(loadIndexList(stateIndices, wrapped->getStateDim(), exclude, "state")),
    keptVelocityIndices(loadIndexList(velocityIndices, wrapped->getVelocityDim(), exclude, "velocity"))
{
}

OMPartial* OMPartial::fromMask(
    ObservationModel* wrapped,
    const std::vector<bool>& mask, bool exclude)
{
    return new OMPartial(wrapped,
                         loadMask(mask, wrapped->getStateDim(), exclude, "state"));
}

OMPartial* OMPartial::fromMask(
    ObservationModel* wrapped, const std::vector<bool>& stateMask, const std::vector<bool>& velocityMask,
    bool exclude)
{
    return new OMPartial(wrapped,
                         loadMask(stateMask, wrapped->getStateDim(), exclude, "state"),
                         loadMask(velocityMask, wrapped->getVelocityDim(), exclude, "velocity"));
}

OMPartial *OMPartial::fromNames(ObservationModel *wrapped, const std::vector<std::string> &names, bool exclude,
                                bool autoSelectVelocity) {
    IndexList states;
    IndexList velocities;

    auto stateNames = wrapped->getStateNames();
    auto velocityNames = wrapped->getVelocityNames();

    for (auto& name : names) {
        // find in state
        auto stateIter = std::find(stateNames.begin(), stateNames.end(), name);
        if (stateIter != stateNames.end()) {
            // add to index list
            states.push_back(std::distance(stateNames.begin(), stateIter));

            if (autoSelectVelocity) {
                // select corresponding velocity
                auto velIter = std::find(velocityNames.begin(), velocityNames.end(), name+"d");
                if (velIter != velocityNames.end()) {
                    // add to index list
                    velocities.push_back(std::distance(velocityNames.begin(), velIter));
                }
            }
        } else {
            // find in velocity
            auto velIter = std::find(velocityNames.begin(), velocityNames.end(), name);
            if (velIter != velocityNames.end()) {
                // add to index list
                velocities.push_back(std::distance(velocityNames.begin(), velIter));
            } else {
                // not found
                std::ostringstream os;
                os << (exclude ? "Excluded " : "Selected ") << " name " << name
                   << " was not found in the wrapped observation model";
                throw std::invalid_argument(os.str());
            }
        }
    }
    return new OMPartial(wrapped, states, velocities, exclude);
}

OMPartial::~OMPartial()
{
    delete wrapped;
}


// Fill partial with the selected values from full
template<typename T>
static void apply(T& partial, const T& full, const std::vector<unsigned int>& keptIndices)
{
    for (unsigned int i = 0; i < keptIndices.size(); ++i) {
        partial[i] = full[keptIndices[i]];
    }
}

unsigned int OMPartial::getStateDim() const
{
    return keptStateIndices.size();
}

unsigned int OMPartial::getVelocityDim() const
{
    return keptVelocityIndices.size();
}

void OMPartial::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    // Allocate temp storage for full, using matnd for potential stack optimization
    MatNd* state_full = NULL;
    MatNd* velocity_full = NULL;
    
    MatNd_create2(state_full, wrapped->getStateDim(), 1);
    MatNd_create2(velocity_full, std::max(wrapped->getVelocityDim(), 1u), 1);
    
    // Retrieve from wrapped
    wrapped->computeObservation(state_full->ele, velocity_full->ele, currentAction, dt);
    
    // Apply selection
    apply(state, state_full->ele, keptStateIndices);
    apply(velocity, velocity_full->ele, keptVelocityIndices);
    
    // Clean up potential allocated memory
    MatNd_destroy(state_full);
    MatNd_destroy(velocity_full);
}

void OMPartial::getLimits(
    double* minState, double* maxState,
    double* maxVelocity) const
{
    // Allocate temp storage for full, using matnd for potential stack optimization
    MatNd* minState_full = NULL;
    MatNd_create2(minState_full, wrapped->getStateDim(), 1);
    MatNd* maxState_full = NULL;
    MatNd_create2(maxState_full, wrapped->getStateDim(), 1);
    MatNd* maxVelocity_full = NULL;
    MatNd_create2(maxVelocity_full, std::max(wrapped->getVelocityDim(), 1u), 1);
    
    // Retrieve from wrapped
    wrapped->getLimits(minState_full->ele, maxState_full->ele, maxVelocity_full->ele);
    
    // Apply selection
    apply(minState, minState_full->ele, keptStateIndices);
    apply(maxState, maxState_full->ele, keptStateIndices);
    apply(maxVelocity, maxVelocity_full->ele, keptVelocityIndices);
    
    // Clean up potential allocated memory
    MatNd_destroy(minState_full);
    MatNd_destroy(maxState_full);
    MatNd_destroy(maxVelocity_full);
}

std::vector<std::string> OMPartial::getStateNames() const
{
    auto full = wrapped->getStateNames();
    std::vector<std::string> partial(keptStateIndices.size());
    apply(partial, full, keptStateIndices);
    return partial;
}

std::vector<std::string> OMPartial::getVelocityNames() const
{
    auto full = wrapped->getVelocityNames();
    std::vector<std::string> partial(keptVelocityIndices.size());
    apply(partial, full, keptVelocityIndices);
    return partial;
}

void OMPartial::reset()
{
    wrapped->reset();
}

std::vector<ObservationModel*> OMPartial::getNested() const
{
    return {wrapped};
}



} /* namespace Rcs */

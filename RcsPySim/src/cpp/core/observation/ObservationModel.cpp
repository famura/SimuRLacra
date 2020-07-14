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

#include "ObservationModel.h"

#include <Rcs_VecNd.h>
#include <Rcs_macros.h>

#include <limits>
#include <sstream>
#include <typeinfo>

namespace Rcs
{

ObservationModel::~ObservationModel() = default;

MatNd* ObservationModel::computeObservation(const MatNd* currentAction, double dt) const
{
    MatNd* result = MatNd_create(getDim(), 1);
    computeObservation(result, currentAction, dt);
    return result;
}

void ObservationModel::computeObservation(MatNd* observation, const MatNd* currentAction, double dt) const
{
    // First state, then velocity
    computeObservation(observation->ele, observation->ele + getStateDim(), currentAction, dt);
}

void ObservationModel::reset()
{
    // Do nothing
}

unsigned int ObservationModel::getVelocityDim() const
{
    // velocity dim == state dim
    return getStateDim();
}

// Set all min-values to -inf and all max-values to +inf by default
void ObservationModel::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    unsigned int sd = getStateDim();
    VecNd_setElementsTo(minState, -std::numeric_limits<double>::infinity(), sd);
    VecNd_setElementsTo(maxState, std::numeric_limits<double>::infinity(), sd);
    VecNd_setElementsTo(maxVelocity, std::numeric_limits<double>::infinity(), getVelocityDim());
}

std::vector<std::string> ObservationModel::getStateNames() const
{
    // Generate default names from class name and numbers
    const char* className = typeid(*this).name();
    
    std::vector<std::string> out;
    for (size_t i = 0; i < getStateDim(); ++i) {
        std::ostringstream os;
        os << className << "_" << i;
        out.push_back(os.str());
    }
    return out;
}

std::vector<std::string> ObservationModel::getVelocityNames() const
{
    // Fast track for no velocities case
    if (getVelocityDim() == 0) {
        return {};
    }
    RCHECK_MSG(getVelocityDim() == getStateDim(),
               "Must override getVelocityNames if velocity dim is not 0 or state dim.");
    
    // Append 'd' to each state name
    std::vector<std::string> out;
    for (auto& stateName : getStateNames()) {
        out.push_back(stateName + "d");
    }
    return out;
}


unsigned int ObservationModel::getDim() const
{
    // Observe state and velocity
    return getStateDim() + getVelocityDim();
}

void ObservationModel::getMinMax(double* min, double* max) const
{
    unsigned int sd = getStateDim();
    // Get the min and max velocity value pointer
    double* minVel = min + sd;
    double* maxVel = max + sd;
    
    // Obtain limits
    getLimits(min, max, maxVel);
    // Derive min velocity from max velocity
    VecNd_constMul(minVel, maxVel, -1, getVelocityDim());
}

std::vector<std::string> ObservationModel::getNames() const
{
    // concat state and velocity names
    auto res = getStateNames();
    auto vn = getVelocityNames();
    std::move(vn.begin(), vn.end(), std::inserter(res, res.end()));
    return res;
}

std::vector<ObservationModel*> ObservationModel::getNested() const
{
    return {};
}

} /* namespace Rcs */

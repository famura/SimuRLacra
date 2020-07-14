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

#include "OMForceTorque.h"

#include <Rcs_typedef.h>
#include <Rcs_math.h>
#include <Rcs_MatNd.h>

#include <limits>
#include <vector>
#include <stdexcept>


namespace Rcs
{

OMForceTorque::OMForceTorque(RcsGraph* graph, const char* sensorName, double max_force) : max_force(max_force)
{
    sensor = RcsGraph_getSensorByName(graph, sensorName);
    if (!sensor) {
        throw std::invalid_argument("Sensor not found: " + std::string(sensorName));
    }
    max_torque = std::numeric_limits<double>::infinity(); // [Nm]
}

OMForceTorque::~OMForceTorque() = default;

unsigned int OMForceTorque::getStateDim() const
{
    return 6;  // 3 forces, 3 torques
}

unsigned int OMForceTorque::getVelocityDim() const
{
    // no derivative
    return 0;
}

void OMForceTorque::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    VecNd_copy(state, sensor->rawData->ele, getStateDim());
}

void OMForceTorque::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    // Forces
    for (size_t i = 0; i < getStateDim()/2; ++i) {
        minState[i] = -max_force;
        maxState[i] = max_force;
    }
    // Torques
    for (size_t i = getStateDim()/2; i < getStateDim(); ++i) {
        minState[i] = -max_torque;
        maxState[i] = max_torque;
    }
}

std::vector<std::string> OMForceTorque::getStateNames() const
{
    std::string sn = sensor->name;
    return {sn + "_Fx", sn + "_Fy", sn + "_Fz", sn + "_Tx", sn + "_Ty", sn + "_Tz",};
}

} /* namespace Rcs */

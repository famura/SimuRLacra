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

#include "ISSBallInTube.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBallInTube::ISSBallInTube(RcsGraph* graph, bool fixedInitState) : InitStateSetter(graph), fixedInitState(fixedInitState)
{
    // Grab direct references to the used bodies
    platform = RcsGraph_getBodyByName(graph, "ImetronPlatform");
    RCHECK(platform);
    rail = RcsGraph_getBodyByName(graph, "RailBot");
    RCHECK(rail);
}

ISSBallInTube::~ISSBallInTube()
{
    // Nothing to destroy
}

unsigned int ISSBallInTube::getDim() const
{
    return 4;  // 3 base, 1 rail
}

void ISSBallInTube::getMinMax(double* min, double* max) const
{
    min[0] = -0.2;  // base_x [m]
    max[0] = +0.2;
    min[1] = -0.05;  // base_y [m]
    max[1] = +0.05;
    min[2] = RCS_DEG2RAD(-5.);  // base_theta [rad]
    max[2] = RCS_DEG2RAD(5.);
    min[3] = 0.8; // rail_z [m]
    max[3] = 0.9;
}

std::vector<std::string> ISSBallInTube::getNames() const
{
    return {"base_x", "base_y", "base_theta", "rail_z"};
}

void ISSBallInTube::applyInitialState(const MatNd* initialState)
{
    bool b0, b1, b2, b3;
    
    // Set the position to the box' rigid body joints directly in global world coordinates
    if (fixedInitState) {
        b0 = RcsGraph_setJoint(graph, "DofBaseX", -0.2);
        b1 = RcsGraph_setJoint(graph, "DofBaseY", 0.);
        b2 = RcsGraph_setJoint(graph, "DofBaseThZ", 0.);
        b3 = RcsGraph_setJoint(graph, "DofChestZ", 0.85);
    }
    else {
        b0 = RcsGraph_setJoint(graph, "DofBaseX", initialState->ele[0]);
        b1 = RcsGraph_setJoint(graph, "DofBaseY", initialState->ele[1]);
        b2 = RcsGraph_setJoint(graph, "DofBaseThZ", initialState->ele[2]);
        b3 = RcsGraph_setJoint(graph, "DofChestZ", initialState->ele[3]);
    }
    
    if (!(b0 && b1 && b2 && b3)) {
        throw std::invalid_argument("Setting graph failed for at least one of the joints!");
    }
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

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

#include "ISSBoxLifting.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxLifting::ISSBoxLifting(RcsGraph* graph, bool fixedInitState) : InitStateSetter(graph),
                                                                     fixedInitState(fixedInitState)
{
    // Grab direct references to the used bodies
    platform = RcsGraph_getBodyByName(graph, "ImetronPlatform");
    RCHECK(platform);
    rail = RcsGraph_getBodyByName(graph, "RailBot");
    RCHECK(rail);
    link2L = RcsGraph_getBodyByName(graph, "lbr_link_2_L");
    RCHECK(link2L);
    link2R = RcsGraph_getBodyByName(graph, "lbr_link_2_R");
    RCHECK(link2R);
}

ISSBoxLifting::~ISSBoxLifting()
{
    // Nothing to destroy
}

unsigned int ISSBoxLifting::getDim() const
{
    return 6;  // 3 base, 1 rail, 2 LBR joints
}

void ISSBoxLifting::getMinMax(double* min, double* max) const
{
    min[0] = 0.05;  // base_x [m]
    max[0] = 0.25;
    min[1] = -0.05;  // base_y [m]
    max[1] = 0.05;
    min[2] = RCS_DEG2RAD(-5.);  // base_theta [rad]
    max[2] = RCS_DEG2RAD(5.);
    min[3] = 0.8; // rail_z [m]
    max[3] = 0.9;
    min[4] = RCS_DEG2RAD(60.); // joint_2_L [rad]
    max[4] = RCS_DEG2RAD(70.);
    min[5] = RCS_DEG2RAD(-70.); // joint_2_R [rad]
    max[5] = RCS_DEG2RAD(-60.);
}

std::vector<std::string> ISSBoxLifting::getNames() const
{
    return {"base_x", "base_y", "base_theta", "rail_z", "joint_2_L", "joint_2_R"};
}

void ISSBoxLifting::applyInitialState(const MatNd* initialState)
{
    bool b0, b1, b2, b3, b4, b5;
    
    // Set the position to the box' rigid body joints directly in global world coordinates
    if (fixedInitState) {
        b0 = RcsGraph_setJoint(graph, "DofBaseX", initialState->ele[0]);
        b1 = RcsGraph_setJoint(graph, "DofBaseY", initialState->ele[1]);
        b2 = RcsGraph_setJoint(graph, "DofBaseThZ", initialState->ele[2]);
        b3 = RcsGraph_setJoint(graph, "DofChestZ", initialState->ele[3]);
        b4 = RcsGraph_setJoint(graph, "lbr_joint_2_L", initialState->ele[4]);
        b5 = RcsGraph_setJoint(graph, "lbr_joint_2_R", initialState->ele[5]);
    }
    else {
        b0 = RcsGraph_setJoint(graph, "DofBaseX", 0.2);
        b1 = RcsGraph_setJoint(graph, "DofBaseY", 0.);
        b2 = RcsGraph_setJoint(graph, "DofBaseThZ", 0.);
        b3 = RcsGraph_setJoint(graph, "DofChestZ", 0.85);
        b4 = RcsGraph_setJoint(graph, "lbr_joint_2_L", RCS_DEG2RAD(65.));
        b5 = RcsGraph_setJoint(graph, "lbr_joint_2_R", RCS_DEG2RAD(-65.));
    }
    
    if (!(b0 && b1 && b2 && b3 && b4 && b5)) {
        throw std::invalid_argument("Setting graph failed for at least one of the joints!");
    }
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

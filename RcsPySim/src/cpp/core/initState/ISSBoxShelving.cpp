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

#include "ISSBoxShelving.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxShelving::ISSBoxShelving(RcsGraph* graph, bool fixedInitState) : InitStateSetter(graph),
                                                                       fixedInitState(fixedInitState)
{
    // Grab direct references to the used bodies
    platform = RcsGraph_getBodyByName(graph, "ImetronPlatform");
    RCHECK(platform);
    rail = RcsGraph_getBodyByName(graph, "RailBot");
    RCHECK(rail);
    link2L = RcsGraph_getBodyByName(graph, "lbr_link_2_L");
    RCHECK(link2L);
    link4L = RcsGraph_getBodyByName(graph, "lbr_link_4_L");
    RCHECK(link4L);
}

ISSBoxShelving::~ISSBoxShelving()
{
    // Nothing to destroy
}

unsigned int ISSBoxShelving::getDim() const
{
    return 6;  // 3 base, 1 rail, 2 LBR joints
}

void ISSBoxShelving::getMinMax(double* min, double* max) const
{
    min[0] = -0.1;  // base X
    max[0] = 0.1;
    min[1] = -0.1;  // base y
    max[1] = 0.1;
    min[2] = RCS_DEG2RAD(-10.);  // base theta z
    max[2] = RCS_DEG2RAD(10.);
    min[3] = 0.7; // rail z
    max[3] = 0.9;
    min[4] = RCS_DEG2RAD(20.); // joint 2
    max[4] = RCS_DEG2RAD(60.);
    min[5] = RCS_DEG2RAD(70.); // joint 4
    max[5] = RCS_DEG2RAD(95.);
}

std::vector<std::string> ISSBoxShelving::getNames() const
{
    return {"base_x", "base_y", "base_theta", "rail_z", "joint_2_L", "joint_4_L"};
}

void ISSBoxShelving::applyInitialState(const MatNd* initialState)
{
    bool b0, b1, b2, b3, b4, b5;
    
    // Set the position to the box' rigid body joints directly in global world coordinates
    if (fixedInitState) {
        b0 = RcsGraph_setJoint(graph, "DofBaseX", 0.);
        b1 = RcsGraph_setJoint(graph, "DofBaseY", 0.);
        b2 = RcsGraph_setJoint(graph, "DofBaseThZ", 0.);
        b3 = RcsGraph_setJoint(graph, "DofChestZ", 0.8);
        b4 = RcsGraph_setJoint(graph, "lbr_joint_2_L", RCS_DEG2RAD(30));
        b5 = RcsGraph_setJoint(graph, "lbr_joint_4_L", RCS_DEG2RAD(90));
    }
    else {
        b0 = RcsGraph_setJoint(graph, "DofBaseX", initialState->ele[0]);
        b1 = RcsGraph_setJoint(graph, "DofBaseY", initialState->ele[1]);
        b2 = RcsGraph_setJoint(graph, "DofBaseThZ", initialState->ele[2]);
        b3 = RcsGraph_setJoint(graph, "DofChestZ", initialState->ele[3]);
        b4 = RcsGraph_setJoint(graph, "lbr_joint_2_L", initialState->ele[4]);
        b5 = RcsGraph_setJoint(graph, "lbr_joint_4_L", initialState->ele[5]);
    }
    
    if (!(b0 && b1 && b2 && b3 && b4 && b5)) {
        throw std::invalid_argument("Setting graph failed for at least one of the joints!");
    }
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

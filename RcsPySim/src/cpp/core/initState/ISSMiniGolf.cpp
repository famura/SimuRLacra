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

#include "ISSMiniGolf.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSMiniGolf::ISSMiniGolf(RcsGraph* graph, bool fixedInitState) : InitStateSetter(graph), fixedInitState(fixedInitState)
{
    // Grab direct references to the used bodies
    ball = RcsGraph_getBodyByName(graph, "Ball");
    RCHECK(ball);
}

ISSMiniGolf::~ISSMiniGolf()
{
    // Nothing to destroy
}

unsigned int ISSMiniGolf::getDim() const
{
    return 9;  // ball x and y position + 7 joint angular positions
}

void ISSMiniGolf::getMinMax(double* min, double* max) const
{
    double ballPosHalfSpanX = 1e-6; // [m]
    double ballPosHalfSpanY = 1e-2; // [m]
    double jointAngHalfSpan = 1e-6; // [rad]
    
    min[0] = 0.304096 - ballPosHalfSpanX;  // ball_x [m]
    max[0] = 0.304096 + ballPosHalfSpanX;
    min[1] = 1.29785 - ballPosHalfSpanY;  // ball_y [m]
    max[1] = 1.29785 + ballPosHalfSpanY;
    min[2] = RCS_DEG2RAD(18.996253 - jointAngHalfSpan);  // base-m3 [rad]
    max[2] = RCS_DEG2RAD(18.996253 + jointAngHalfSpan);
    min[3] = RCS_DEG2RAD(-87.227101 - jointAngHalfSpan);  // m3-m4 [rad]
    max[3] = RCS_DEG2RAD(-87.227101 + jointAngHalfSpan);
    min[4] = RCS_DEG2RAD(74.149568 - jointAngHalfSpan);  // m4-m5 [rad]
    max[4] = RCS_DEG2RAD(74.149568 + jointAngHalfSpan);
    min[5] = RCS_DEG2RAD(-75.577025 - jointAngHalfSpan);  // m5-m6 [rad]
    max[5] = RCS_DEG2RAD(-75.577025 + jointAngHalfSpan);
    min[6] = RCS_DEG2RAD(56.207369 - jointAngHalfSpan);  // m6-m7 [rad]
    max[6] = RCS_DEG2RAD(56.207369 + jointAngHalfSpan);
    min[7] = RCS_DEG2RAD(-175.162794 - jointAngHalfSpan);  // m7-m8 [rad]
    max[7] = RCS_DEG2RAD(-175.162794 + jointAngHalfSpan);
    min[8] = RCS_DEG2RAD(-41.543793 - jointAngHalfSpan);  // m8-m9 [rad]
    max[8] = RCS_DEG2RAD(-41.543793 + jointAngHalfSpan);
}

std::vector<std::string> ISSMiniGolf::getNames() const
{
    return {"ball_x", "ball_y", "base-m3", "m3-m4", "m4-m5", "m5-m6", "m6-m7", "m7-m8", "m8-m9"};
}

void ISSMiniGolf::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    double* ballRBJ = &graph->q->ele[ball->jnt->jointIndex];
    if (fixedInitState) {
        ballRBJ[0] = 0.304096; // ball_x [m]
        ballRBJ[1] = 1.29785; // ball_y [m]
        RcsGraph_setJoint(graph, "base-m3", RCS_DEG2RAD(18.996253));
        RcsGraph_setJoint(graph, "m3-m4", RCS_DEG2RAD(-87.227101));
        RcsGraph_setJoint(graph, "m4-m5", RCS_DEG2RAD(74.149568));
        RcsGraph_setJoint(graph, "m5-m6", RCS_DEG2RAD(-75.577025));
        RcsGraph_setJoint(graph, "m6-m7", RCS_DEG2RAD(56.207369));
        RcsGraph_setJoint(graph, "m7-m8", RCS_DEG2RAD(-175.162794));
        RcsGraph_setJoint(graph, "m8-m9", RCS_DEG2RAD(-41.543793));
    }
    else {
        ballRBJ[0] = initialState->ele[0];
        ballRBJ[1] = initialState->ele[1];
        RcsGraph_setJoint(graph, "base-m3", initialState->ele[2]);
        RcsGraph_setJoint(graph, "m3-m4", initialState->ele[3]);
        RcsGraph_setJoint(graph, "m4-m5", initialState->ele[4]);
        RcsGraph_setJoint(graph, "m5-m6", initialState->ele[5]);
        RcsGraph_setJoint(graph, "m6-m7", initialState->ele[6]);
        RcsGraph_setJoint(graph, "m7-m8", initialState->ele[7]);
        RcsGraph_setJoint(graph, "m8-m9", initialState->ele[8]);
    }
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

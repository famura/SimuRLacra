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
    return 2;  // ball x and y position
}

void ISSMiniGolf::getMinMax(double* min, double* max) const
{
    min[0] = 0.48;  // base_x [m]
    max[0] = 0.52;
    min[1] = 1.3;  // base_y [m]
    max[1] = 1.5;
}

std::vector<std::string> ISSMiniGolf::getNames() const
{
    return {"ball_x", "ball_y"};
}

void ISSMiniGolf::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    double* ballRBJ = &graph->q->ele[ball->jnt->jointIndex];
    if (fixedInitState) {
        ballRBJ[0] = 0.5; // ball_x [m]
        ballRBJ[1] = 1.4; // ball_y [m]
    }
    else {
        ballRBJ[0] = initialState->ele[0];
        ballRBJ[1] = initialState->ele[1];
    }
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

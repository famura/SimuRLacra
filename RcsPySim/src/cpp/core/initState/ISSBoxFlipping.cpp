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

#include "ISSBoxFlipping.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxFlipping::ISSBoxFlipping(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    wrist1L = RcsGraph_getBodyByName(graph, "Wrist1_L");
    RCHECK(wrist1L);
    wrist2L = RcsGraph_getBodyByName(graph, "Wrist2_L");
    RCHECK(wrist2L);
    wrist3L = RcsGraph_getBodyByName(graph, "Wrist3_L");
    RCHECK(wrist3L);
    wrist1R = RcsGraph_getBodyByName(graph, "Wrist1_R");
    RCHECK(wrist1R);
    wrist2R = RcsGraph_getBodyByName(graph, "Wrist2_R");
    RCHECK(wrist2R);
    wrist3R = RcsGraph_getBodyByName(graph, "Wrist3_R");
    RCHECK(wrist3R);
}

ISSBoxFlipping::~ISSBoxFlipping()
{
    // Nothing to destroy
}

unsigned int ISSBoxFlipping::getDim() const
{
    return 6;
}

void ISSBoxFlipping::getMinMax(double* min, double* max) const
{
    min[0] = 1.25;
    max[0] = 1.25;
    min[1] = 0.2;
    max[1] = 0.2;
    min[2] = 0.95;
    max[2] = 0.95;
    min[3] = 1.25;
    max[3] = 1.25;
    min[4] = -0.2;
    max[4] = -0.2;
    min[5] = 0.95;
    max[5] = 0.95;
}

std::vector<std::string> ISSBoxFlipping::getNames() const
{
    return {"x_L", "y_L", "z_L", "x_L", "y_L", "z_L"};
}

void ISSBoxFlipping::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    graph->q->ele[wrist1L->jnt->jointIndex] = initialState->ele[0];
    graph->q->ele[wrist2L->jnt->jointIndex] = initialState->ele[1];
    graph->q->ele[wrist3L->jnt->jointIndex] = initialState->ele[2];
    graph->q->ele[wrist1R->jnt->jointIndex] = initialState->ele[3];
    graph->q->ele[wrist2R->jnt->jointIndex] = initialState->ele[4];
    graph->q->ele[wrist3R->jnt->jointIndex] = initialState->ele[5];
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

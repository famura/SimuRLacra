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

#include "ISSPlanar3Link.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSPlanar3Link::ISSPlanar3Link(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    link1 = RcsGraph_getBodyByName(graph, "Link1");
    link2 = RcsGraph_getBodyByName(graph, "Link2");
    link3 = RcsGraph_getBodyByName(graph, "Link3");
    RCHECK(link1);
    RCHECK(link2);
    RCHECK(link3);
}

ISSPlanar3Link::~ISSPlanar3Link()
{
    // Nothing to destroy
}

unsigned int ISSPlanar3Link::getDim() const
{
    return 3;
}

void ISSPlanar3Link::getMinMax(double* min, double* max) const
{
    // Joint angles [rad] (velocity stays on default)
    min[0] = RCS_DEG2RAD(10.);
    max[0] = RCS_DEG2RAD(10.);
    min[1] = RCS_DEG2RAD(10.);
    max[1] = RCS_DEG2RAD(10.);
    min[2] = RCS_DEG2RAD(10.);
    max[2] = RCS_DEG2RAD(10.);
}

std::vector<std::string> ISSPlanar3Link::getNames() const
{
    return {"q1", "q2", "q3"};
}

void ISSPlanar3Link::applyInitialState(const MatNd* initialState)
{
    // Get the relative joint angles
    double q1_init = initialState->ele[0];
    double q2_init = initialState->ele[1];
    double q3_init = initialState->ele[2];
    
    // Set the position to the links's rigid body joints
    graph->q->ele[link1->jnt->jointIndex] = q1_init;
    graph->q->ele[link2->jnt->jointIndex] = q2_init;
    graph->q->ele[link3->jnt->jointIndex] = q3_init;
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}
    
} /* namespace Rcs */

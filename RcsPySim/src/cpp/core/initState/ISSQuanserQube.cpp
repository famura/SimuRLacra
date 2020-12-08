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

#include "ISSQuanserQube.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSQuanserQube::ISSQuanserQube(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    arm = RcsGraph_getBodyByName(graph, "Arm");
    RCHECK (arm);
    pendulum = RcsGraph_getBodyByName(graph, "Pendulum");
    RCHECK (pendulum);
}

ISSQuanserQube::~ISSQuanserQube()
{
    // nothing to destroy
}

unsigned int ISSQuanserQube::getDim() const
{
    return 4;
}

std::vector<std::string> ISSQuanserQube::getNames() const
{
    return {"theta", "alpha", "theta_dot", "alpha_dot"};
}

void ISSQuanserQube::getMinMax(double* min, double* max) const
{
    min[0] = RCS_DEG2RAD(-180.);  // arm angle [rad]
    max[0] = RCS_DEG2RAD(180.);
    min[1] = RCS_DEG2RAD(-180.); // pendulum angle [rad]
    max[1] = RCS_DEG2RAD(180.);
    min[2] = RCS_DEG2RAD(-10.); // arm velocity in [rad/s]
    max[2] = RCS_DEG2RAD(10);
    min[3] = RCS_DEG2RAD(-10);  // pendulum velocity [rad/s]
    max[3] = RCS_DEG2RAD(10);
}

void ISSQuanserQube::applyInitialState(const MatNd* initialState)
{
    // The initialState is provided in rad
    
    // Set the angular position to the arm's and the pendulum's rigid body joints
    // graph->q is a vector of dim 2x1
    graph->q->ele[arm->jnt->jointIndex] = initialState->ele[0];
    graph->q->ele[pendulum->jnt->jointIndex] = initialState->ele[1];
    
    // Set the angular velocity to the arm's and the pendulum's rigid body joints
    // graph->q is a vector of dim 2x1
    graph->q_dot->ele[arm->jnt->jointIndex] = initialState->ele[2];
    graph->q_dot->ele[pendulum->jnt->jointIndex] = initialState->ele[3];
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

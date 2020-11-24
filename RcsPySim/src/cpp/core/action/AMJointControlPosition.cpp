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

#include "AMJointControlPosition.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>

#include <iostream>

namespace Rcs
{


AMJointControlPosition::AMJointControlPosition(RcsGraph* graph) : AMJointControl(graph)
{
    // Make sure nJ is correct
    RcsGraph_setState(graph, NULL, NULL);
    // Iterate over unconstrained joints
    REXEC(1) {
        RCSGRAPH_TRAVERSE_JOINTS(graph) {
                if (JNT->jacobiIndex != -1) {
                    // Check if the joints actually use position control inside the simulation
                    if (JNT->ctrlType != RCSJOINT_CTRL_POSITION) {
                        std::cout
                            << "Using AMJointControlPosition, but at least one joint does not have the control type "
                               "RCSJOINT_CTRL_POSITION!" << std::endl;
                    }
                }
            }
    }
}

AMJointControlPosition::~AMJointControlPosition()
{
    // Nothing to destroy
}

void
AMJointControlPosition::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    RcsGraph_stateVectorFromIK(graph, action, q_des);
}

void AMJointControlPosition::getMinMax(double* min, double* max) const
{
    RCSGRAPH_TRAVERSE_JOINTS(graph) {
            if (JNT->jacobiIndex != -1) {
                // Set min/max from joint limits
                min[JNT->jacobiIndex] = JNT->q_min;
                max[JNT->jacobiIndex] = JNT->q_max;
            }
        }
}

void AMJointControlPosition::getStableAction(MatNd* action) const
{
    // Stable action = current state
    RcsGraph_stateVectorToIK(graph, graph->q, action);
}

std::vector<std::string> AMJointControlPosition::getNames() const
{
    std::vector<std::string> out;
    RCSGRAPH_TRAVERSE_JOINTS(graph) {
            if (JNT->jacobiIndex != -1) {
                out.emplace_back(JNT->name);
            }
        }
    
    return out;
}

ActionModel* AMJointControlPosition::clone(RcsGraph* newGraph) const
{
    return new AMJointControlPosition(newGraph);
}

} /* namespace Rcs */


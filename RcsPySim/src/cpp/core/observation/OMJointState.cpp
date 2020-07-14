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

#define _USE_MATH_DEFINES

#include "OMJointState.h"
#include "OMCombined.h"

#include <Rcs_typedef.h>
#include <Rcs_joint.h>
#include <Rcs_math.h>

#include <sstream>
#include <stdexcept>
#include <cmath>

namespace Rcs
{


static bool defaultWrapJointAngle(RcsJoint* joint)
{
    // Wrap if it models one full rotation
    return RcsJoint_isRotation(joint) && joint->q_min == -M_PI && joint->q_max == M_PI;
}

OMJointState::OMJointState(RcsGraph* graph, const char* jointName, bool wrapJointAngle) :
    graph(graph), wrapJointAngle(wrapJointAngle)
{
    joint = RcsGraph_getJointByName(graph, jointName);
    if (!joint) {
        std::ostringstream os;
        os << "Unable to find joint " << jointName << " in graph.";
        throw std::invalid_argument(os.str());
    }
    if (wrapJointAngle && !RcsJoint_isRotation(joint)) {
        std::ostringstream os;
        os << "Joint " << jointName << " is not a rotation joint, so we cannot wrap the joint angle.";
        throw std::invalid_argument(os.str());
    }
}

OMJointState::OMJointState(RcsGraph* graph, const char* jointName) : OMJointState(graph, jointName, false)
{
    wrapJointAngle = defaultWrapJointAngle(joint);
}

OMJointState::OMJointState(RcsGraph* graph, RcsJoint* joint) :
    graph(graph),
    joint(joint),
    wrapJointAngle(defaultWrapJointAngle(joint))
{
}

OMJointState::~OMJointState()
{
    // Nothing else to destroy
}


unsigned int OMJointState::getStateDim() const
{
    return 1;
}

void OMJointState::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    double q = graph->q->ele[joint->jointIndex];
    if (wrapJointAngle) {
        q = Math_fmodAngle(q);
    }
    *state = q;
    *velocity = graph->q_dot->ele[joint->jointIndex];
}

void OMJointState::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    // Use joint limits from graph (in contrast to the other observation models)
    *minState = joint->q_min;
    *maxState = joint->q_max;
    *maxVelocity = joint->speedLimit;
}

std::vector<std::string> OMJointState::getStateNames() const
{
    return {joint->name};
}

ObservationModel* OMJointState::observeAllJoints(RcsGraph* graph)
{
    auto combined = new OMCombined();
    RCSGRAPH_TRAVERSE_JOINTS(graph) {
            combined->addPart(new OMJointState(graph, JNT));
        }
    return combined;
}

ObservationModel* OMJointState::observeUnconstrainedJoints(RcsGraph* graph)
{
    auto combined = new OMCombined();
    RCSGRAPH_TRAVERSE_JOINTS(graph) {
            if (!JNT->constrained) {
                combined->addPart(new OMJointState(graph, JNT));
            }
        }
    return combined;
}

} /* namespace Rcs */

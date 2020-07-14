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

#include "AMJointControlAcceleration.h"

#include <Rcs_typedef.h>
#include <Rcs_dynamics.h>
#include <Rcs_macros.h>
#include <Rcs_VecNd.h>

#include <limits>


namespace Rcs
{

// TODO @Felix: this does not work ATM, but I will leave it in case we want to change that later.
AMJointControlAcceleration::AMJointControlAcceleration(RcsGraph* graph) : AMJointControl(graph)
{
    // Make sure nJ is correct
    RcsGraph_setState(graph, NULL, NULL);
    // Iterate unconstrained joints
//    RCSGRAPH_TRAVERSE_JOINTS(graph)
//    {
//        if (JNT->jacobiIndex != -1)
//        {
//            // Make sure that the joints actually use torque control inside the simulation
//            JNT->ctrlType = RCSJOINT_CTRL_TORQUE;
//        }
//    }
    
    // create temporary matrices
    M = MatNd_create(graph->nJ, graph->nJ);
    h = MatNd_create(graph->nJ, 1);
    F_gravity = MatNd_create(graph->nJ, 1);
}

AMJointControlAcceleration::~AMJointControlAcceleration()
{
    MatNd_destroy(M);
    MatNd_destroy(h);
    MatNd_destroy(F_gravity);
}

// TODO discuss this fcn with Michael
void AMJointControlAcceleration::computeCommand(
    MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action,
    double dt)
{
    // Compute inverse dynamics
    RcsGraph_computeKineticTerms(graph, M, h, F_gravity);
    
    // Use them to compute the torque. This is done in IK state vector.
    MatNd_reshape(T_des, graph->nJ, 1);
    
    MatNd_mul(T_des, M, action);
    
    // Print if debug level is exceeded
    REXEC(8) {
        MatNd_printComment("M", M);
        MatNd_printComment("M*q_dotdot_des", T_des);
        MatNd_printComment("coriolis", h);
        MatNd_printComment("gravity compensation", F_gravity);
    }
    
    MatNd_addSelf(T_des, h);
    MatNd_subSelf(T_des, F_gravity);
    // Expand T_des to full state vector
    RcsGraph_stateVectorFromIKSelf(graph, T_des);
}

void AMJointControlAcceleration::getMinMax(double* min, double* max) const
{
    VecNd_setElementsTo(min, -std::numeric_limits<double>::infinity(), getDim()); // rad/s^2
    VecNd_setElementsTo(max, std::numeric_limits<double>::infinity(), getDim()); // rad/s^2
}

void AMJointControlAcceleration::getStableAction(MatNd* action) const
{
    // stable action = no acceleration
    MatNd_setZero(action);
}

ActionModel* AMJointControlAcceleration::clone(RcsGraph* newGraph) const
{
    return new AMJointControlAcceleration(newGraph);
}

} /* namespace Rcs */

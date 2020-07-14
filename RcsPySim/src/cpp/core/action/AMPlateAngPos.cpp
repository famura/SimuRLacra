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

#include "AMPlateAngPos.h"

#include <Rcs_macros.h>
//#include <TaskPose6D.h>
#include <TaskPosition3D.h>
#include <TaskGenericEuler3D.h>

namespace Rcs
{

AMPlateAngPos::AMPlateAngPos(RcsGraph* graph) : ActionModelIK(graph)
{
    // lookup plate body on desired state graph
    RcsBody* plate = RcsGraph_getBodyByName(desiredGraph, "Plate");
    RCHECK(plate);
    // add the dynamicalSystems
    addTask(new TaskPosition3D(desiredGraph, plate, NULL, NULL));
    addTask(new TaskGenericEuler3D(desiredGraph, "CABr", plate, NULL, NULL));
    
    // create state matrix
    x_des = MatNd_create(getController()->getTaskDim(), 1);
    // init state with current
    getController()->computeX(x_des);
}

AMPlateAngPos::~AMPlateAngPos()
{
    MatNd_destroy(x_des);
}

unsigned int AMPlateAngPos::getDim() const
{
    return 2;
}

void AMPlateAngPos::getMinMax(double* min, double* max) const
{
    double maxAngle = 45*M_PI/180;
    min[0] = -maxAngle;
    min[1] = -maxAngle;
    max[0] = maxAngle;
    max[1] = maxAngle;
}

std::vector<std::string> AMPlateAngPos::getNames() const
{
    return {"a", "b"};
}

void AMPlateAngPos::computeCommand(
    MatNd* q_des, MatNd* q_dot_des, MatNd* T_des,
    const MatNd* action, double dt)
{
    // copy actions into relevant parts of x_des
    x_des->ele[4] = action->ele[0]; // alpha
    x_des->ele[5] = action->ele[1]; // beta
    
    // use IK to compute q_des
    computeIK(q_des, q_dot_des, T_des, x_des, dt);
}

void AMPlateAngPos::reset()
{
    ActionModelIK::reset();
    // init state with current
    getController()->computeX(x_des);
}

void AMPlateAngPos::getStableAction(MatNd* action) const
{
    MatNd* x_curr = NULL;
    MatNd_create2(x_curr, getController()->getTaskDim(), 1);
    // compute current state
    getController()->computeX(x_curr);
    // export relevant parts of action
    action->ele[0] = x_curr->ele[4]; // alpha
    action->ele[1] = x_curr->ele[5]; // beta
    // cleanup
    MatNd_destroy(x_curr);
}

ActionModel* AMPlateAngPos::clone(RcsGraph* newGraph) const
{
    auto res = new AMPlateAngPos(newGraph);
    res->setupCollisionModel(collisionMdl);
    return res;
}

} /* namespace Rcs */


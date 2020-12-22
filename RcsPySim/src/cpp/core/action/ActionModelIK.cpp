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

#include "ActionModelIK.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_kinematics.h>
#include <Rcs_basicMath.h>
#include <IkSolverConstraintRMR.h>


/**
 Value freeRatio is the ratio of the joint's half range in which the joint limit gradient kicks in. For instance if
 freeRatio is 1, the whole joint range is without gradient. If freeRatio is 0, the gradient follows a quadratic cost
 over the whole range.

 Value freeRange is the range starting from q0, where the gradient is 0. The value penaltyRange is the range where
 the gradient is not 0. Here is an illustration:

 q_lowerLimit       q_lowerRange     q_0      q_upperRange     q_upperLimit
 |                  |           |             |                |
 |                  |           |             |                |
 \__________________/\__________/\____________/\_______________/
 a               b             c               d

 freeRatio:           b/(a+b)                c/(c+d)
 freeRange:           b                      c
 penaltyRange:        a                      d

 cost =
 q in a: 0.5 * wJL * ((q-q_lowerRange)/a)^2   => c(limit) = 0.5*wJL
 q in b: 0
 q in c: 0
 q in d: 0.5 * wJL * ((q-q_upperRange)/d)^2   => c(limit) = 0.5*wJL

 gradient =
 q in a: (q-q_lowerRange)*wJL/a^2   => grad(limit) = 0.5*wJL/a
 q in b: 0
 q in c: 0
 q in d: (q-q_upperRange)*wJL/a^2   => grad(limit) = 0.5*wJL/d

 For the default case wJL=1 and freeRange=0, we get: dH_i = 0.5/halfRange
 and for symmetric joint centers, we get: dH_i = 1.0/range
 If we interpret it as a velocity, it's the inverse of the joint range.
 */
/*
static void RcsGraph_jointLimitBorderGradient3(const RcsGraph* self, MatNd* dH, double range, double maxSpeedScaling)
{
    RCHECK(range > 0.0);
    RCHECK(maxSpeedScaling > 0.0);
    
    unsigned int dimension = dH->m*dH->n;
    
    RcsStateType type;
    
    if (dimension == self->dof)
    {
        type = RcsStateFull;
    }
    else if (dimension == self->nJ)
    {
        type = RcsStateIK;
    }
    else
    {
        RFATAL("Wrong gradient size: dof=%d  nJ=%d  m=%d  n=%d", self->dof, self->nJ, dH->m, dH->n);
    }
    
    MatNd_reshapeAndSetZero(dH, 1, dimension);
    
    RCSGRAPH_TRAVERSE_JOINTS(self)
        {
            if (JNT->constrained && (type == RcsStateIK))
            {
                continue;
            }
            
            // Coupled joints don't contribute. Limit avoidance is the master joint's job
            if (JNT->coupledTo != nullptr)
            {
                continue;
            }
            
            int index = (type == RcsStateIK) ? JNT->jacobiIndex : JNT->jointIndex;
            double q = Math_clip(self->q->ele[JNT->jointIndex], JNT->q0 - range, JNT->q0 + range);
            dH->ele[index] = maxSpeedScaling*JNT->speedLimit*JNT->weightJL*(q - JNT->q0)/range;
            
        } // RCSGRAPH_TRAVERSE_JOINTS
    
}
*/
namespace Rcs
{
ActionModelIK::ActionModelIK(RcsGraph* graph) : ActionModel(graph), alpha(1e-4), lambda(1e-6)
{
    // Create controller using a separate graph. That graph will later hold the desired state.
    desiredGraph = RcsGraph_clone(graph);
    controller = new ControllerBase(desiredGraph);
    
    // Initialize temp storage where possible
    dq_ref = MatNd_create(graph->dof, 1);
    dq_ref = MatNd_create(graph->dof, 1);
    dH = MatNd_create(1, graph->nJ);
    
    // Since the subclass must initialize the dynamicalSystems first, we defer creation of these to the first reset call.
    solver = nullptr;
    dx_des = nullptr;
    
    // Must be set manually
    collisionMdl = nullptr;
}

ActionModelIK::ActionModelIK(RcsGraph* graph, std::vector<Task*> tasks) : ActionModelIK(graph)
{
    // Populate tasks
    for (auto task : tasks) {
        addTask(task);
    }
    // Create solver eagerly
    reset();
}

ActionModelIK::~ActionModelIK()
{
    MatNd_destroy(dH);
    MatNd_destroy(dx_des);
    MatNd_destroy(dq_ref);
    RcsCollisionModel_destroy(collisionMdl);
    delete solver;
    delete controller;
    // desiredGraph was destroyed by controller
}

void ActionModelIK::addTask(Task* task)
{
    RCHECK_MSG(!solver, "Cannot add a Task to the action model after the first reset() call.");
    // It's pretty easy to create a task using the wrong graph, which leads to all kinds of untraceable errors.
    // So, we use clone to make sure the graph and bodies are the correct instances. We don't need it in all cases,
    // but it also doesn't hurt. Since we officially take ownership of task, we just delete it right after.
    controller->add(task->clone(desiredGraph));
    delete task;
}

void ActionModelIK::reset()
{
    if (solver == nullptr) {
        // Init solver now that the dynamicalSystems are created
        solver = new IkSolverRMR(controller);
        // Init temp storage
        dx_des = MatNd_create(controller->getTaskDim(), 1);
    }
    
    // Updated the controller's graph with the potentially randomized shapes
    RcsGraph_copyResizeableShapes(desiredGraph, graph, 1);
    
    // Copy init state to desired state graph
    RcsGraph_setState(desiredGraph, graph->q, graph->q_dot);
}

void ActionModelIK::computeIK(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* x_des, double dt)
{
    // Compute dx from x_des and x_cur (NOTE: dx is the error, not the time derivative)
    controller->computeDX(dx_des, x_des);
    
    // Compute IK from dx_des
    ikFromDX(q_des, q_dot_des, dt);
}

void ActionModelIK::computeIKVel(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* x_dot_des, double dt)
{
    // Compute dx from x_dot_des. dx is the distance we want to move during this step
    MatNd_constMul(dx_des, x_dot_des, dt);
    
    // Compute IK from dx_des
    ikFromDX(q_des, q_dot_des, dt);
}


void ActionModelIK::ikFromDX(MatNd* q_des, MatNd* q_dot_des, double dt) const
{
    // Print desired task space delta if debug level is exceeded
    REXEC(6) {
        MatNd_printComment("dx_des (from controller->computeDX)", dx_des);
    }
    
    // Compute nullptr space gradients
    RcsGraph_jointLimitGradient(desiredGraph, dH, RcsStateIK);  // more robust choice
//    RcsGraph_jointLimitBorderGradient3(desiredGraph, dH, RCS_DEG2RAD(90.0), 1.0);  // sensitive to range value
    
    // Add collision cost if available
    if (collisionMdl != nullptr) {
        MatNd* dH_ca = MatNd_create(1, controller->getGraph()->nJ);
        // Compute the collision gradient
        RcsCollisionModel_compute(collisionMdl);
        RcsCollisionMdl_gradient(collisionMdl, dH_ca);
        // Add it to nullptrspace
        MatNd_addSelf(dH, dH_ca);
        MatNd_destroy(dH_ca);
    }
    
    // Compute joint space position error with IK
    MatNd_constMulSelf(dH, alpha);
    solver->solveRightInverse(dq_ref, dx_des, dH, nullptr, lambda); // tries to solve everything exactly

//        MatNd* dq_ref_ts = MatNd_create(graph->dof, 1);
//        MatNd* dq_ref_ns = MatNd_create(graph->dof, 1);
//        MatNd* a_temp = MatNd_create(controller->getNumberOfTasks(), 1);
//        solver->solveLeftInverse(dq_ref_ts, dq_ref_ns, dx_des, dH, a_temp, lambda); // tries to solve everything exactly
//        MatNd_add(dq_ref, dq_ref_ts, dq_ref_ns);
//        MatNd_destroy(dq_ref_ts);
//        MatNd_destroy(dq_ref_ns);
//        MatNd_destroy(a_temp);
    
    // Check for speed limit violations
    RcsGraph_limitJointSpeeds(desiredGraph, dq_ref, dt, RcsStateFull);
    
    // Integrate desired joint positions
    MatNd_add(q_des, desiredGraph->q, dq_ref);
    
    // Compute velocity from error
    MatNd_constMul(q_dot_des, dq_ref, 1/dt);
    
    // Update desired state graph for next step
    RcsGraph_setState(desiredGraph, q_des, q_dot_des);
    
    // Also compute gravity compensation force (Bullet might need it, Vortex ignores it)
    // RcsGraph_computeGravityTorque(desiredGraph, T_des);
    // MatNd_constMulSelf(T_des, -1.0);
}

RcsGraph* ActionModelIK::getDesiredGraph() const
{
    return desiredGraph;
}

void ActionModelIK::setupCollisionModel(const RcsCollisionMdl* modelToCopy)
{
    // Copy collision model for desired graph
    if (modelToCopy != nullptr) {
        collisionMdl = RcsCollisionModel_clone(modelToCopy, desiredGraph);
    }
    else {
        collisionMdl = nullptr;
    }
}

/*
 * AMIKGeneric
 */
unsigned int AMIKGeneric::getDim() const
{
    return controller->getTaskDim();
}

void AMIKGeneric::getMinMax(double* min, double* max) const
{
    unsigned int idx = 0;
    for (unsigned int ti = 0; ti < controller->getNumberOfTasks(); ++ti) {
        Task* task = controller->getTask(ti);
        for (unsigned int tp = 0; tp < task->getDim(); ++tp) {
            auto param = task->getParameter(tp);
            min[idx] = param.minVal;
            max[idx] = param.maxVal;
            idx++;
        }
    }
}

std::vector<std::string> AMIKGeneric::getNames() const
{
    std::vector<std::string> result;
    for (unsigned int ti = 0; ti < controller->getNumberOfTasks(); ++ti) {
        Task* task = controller->getTask(ti);
        for (unsigned int tp = 0; tp < task->getDim(); ++tp) {
            auto param = task->getParameter(tp);
            result.push_back(param.name);
        }
    }
    return result;
}

void AMIKGeneric::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    // Copy the ExperimentConfig graph which has been updated by the physics simulation into the desired graph
    RcsGraph_copyRigidBodyDofs(desiredGraph->q, graph, nullptr);
    
    computeIK(q_des, q_dot_des, T_des, action, dt);
}

void AMIKGeneric::getStableAction(MatNd* action) const
{
    controller->computeX(action);
}

ActionModel* AMIKGeneric::clone(RcsGraph* newGraph) const
{
    auto res = new AMIKGeneric(newGraph);
    for (auto task : controller->getTasks()) {
        res->addTask(task->clone(newGraph));
    }
    res->setupCollisionModel(collisionMdl);
    return res;
}
    
} /* namespace Rcs */

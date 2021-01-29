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

#include "AMIKControllerActivation.h"
#include "ActionModelIK.h"

#include <Rcs_macros.h>
#include <Rcs_basicMath.h>

#include <sstream>

namespace Rcs
{

void MatNd_fabsClipEleSelf(MatNd* self, double negUpperBound, double posLowerBound)
{
    unsigned int i, mn = self->m*self->n;
    for (i = 0; i < mn; i++) {
        if (self->ele[i] < 0){
            self->ele[i] = std::min(self->ele[i], negUpperBound);
        }
        else if (self->ele[i] > 0){
            self->ele[i] = std::max(self->ele[i], posLowerBound);
        }
        else{
            self->ele[i] = posLowerBound; // arbitrary choice
        }
    }
}

MatNd* MatNd_signs(const MatNd* src)
{
    MatNd* signs = MatNd_createLike(src);
    for (unsigned int i = 0; i < src->m; i++) {
        for (unsigned int j = 0; j < src->n; j++) {
            double val = (MatNd_get(src, i, j) >= 0) ? 1.0 : -1.0;
            MatNd_set(signs, i, j, val);
        }
    }
    return signs;
}


AMIKControllerActivation::AMIKControllerActivation(RcsGraph* graph, TaskCombinationMethod tcm) :
    AMIKGeneric(graph), taskCombinationMethod(tcm)
{
    this->x_des = MatNd_clone(dx_des); // dx_des comes from AMIKGeneric
    activation = MatNd_create((unsigned int) getController()->getNumberOfTasks(), 1);
    dimAlwaysActiveTasks = 0;
}

AMIKControllerActivation::~AMIKControllerActivation()
{
    delete x_des;
    delete activation;
}

void AMIKControllerActivation::addAlwaysActiveTask(Task* task)
{
    dimAlwaysActiveTasks += task->getDim();
    this->addTask(task);
}

unsigned int AMIKControllerActivation::getDim() const
{
    return (unsigned int) getController()->getNumberOfTasks() - dimAlwaysActiveTasks;
}

void AMIKControllerActivation::getMinMax(double* min, double* max) const
{
    // All activations are between 0 and 1
    for (unsigned int i = 0; i < getDim(); i++) {
        min[i] = -1;
        max[i] = 1;
    }
}

std::vector<std::string> AMIKControllerActivation::getNames() const
{
    std::vector<std::string> names;
    for (unsigned int i = 0; i < getDim(); ++i) {
        names.push_back("a_" + std::to_string(i));
    }
    return names;
}

void AMIKControllerActivation::reset()
{
    ActionModelIK::reset();
    
    activation = MatNd_create((unsigned int) getController()->getNumberOfTasks(), 1);
}

void AMIKControllerActivation::computeCommand(
    MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    RCHECK(action->n == 1);  // actions are column vectors
    
    // Copy the ExperimentConfig graph which has been updated by the physics simulation into the desired graph
    RcsGraph_copyRigidBodyDofs(desiredGraph->q, graph, nullptr);
    
    // Combine the individual activations of every controller task
    MatNd* action_tmp; // working copy
    MatNd* action_abs; // workaround for negative actions
//    MatNd* action_signs; // workaround for negative actions
    action_tmp = MatNd_clone(action);
    action_abs = MatNd_clone(action);
    MatNd_fabsEleSelf(action_abs);
//    action_signs = MatNd_signs(action);
    switch (taskCombinationMethod) {
        case TaskCombinationMethod::Sum:
            break; // no weighting
        
        case TaskCombinationMethod::Mean: {
            MatNd_constMulSelf(action_tmp, 1./MatNd_sumEle(action_abs));
            break;
        }
        
        case TaskCombinationMethod::SoftMax: {
            MatNd_softMax(action_tmp, action, action->m);  // action->m is a neat heuristic for beta
            break;
        }
        
        case TaskCombinationMethod::Product: {
            for (unsigned int i = 0; i < action->m; i++) {
                // Create temp matrix
                MatNd* otherActions; // other actions are all actions without the current
                MatNd_clone2(otherActions, action);
                MatNd_deleteRow(otherActions, i);
                
                // Treat the actions as probability of events and compute the probability that all other actions are false
                double prod = 1;  // 1 is the neutral element for multiplication
                for (unsigned int idx = 0; idx < otherActions->m; idx++) {
                    REXEC(7) {
                        std::cout << "factor " << (1 - otherActions->ele[idx]) << std::endl;
                    }
                    // One part of the product is always 1
                    prod *= (1 - otherActions->ele[idx]);
                }
                MatNd_destroy(otherActions);
                
                REXEC(7) {
                    std::cout << "prod " << prod << std::endl;
                }
                
                MatNd_set(action_tmp, i, 0, action->ele[i]*prod);
            }
            
            break;
        }
    }
    
    // Stabilize actions. If an action is exactly zero, computeDX will discard that task, leading to a shape error.
    if (MatNd_minEle(action_abs) < 1e-6) {
        MatNd_fabsClipEleSelf(action_tmp, -1e6, 1e6);
        REXEC(5) {
            std::cout << "Clipped the activations to ]-inf, -1e-6] u [1e-6, inf[" << std::endl;
        }
    }
    
    // Fill the first rows with the variable activations and the remaining rows with ones for the always active tasks
    MatNd_copyRows(activation, 0, action_tmp, 0, action_tmp->m);
    for (unsigned int i = action_tmp->m; i < activation->m; i++) {
        activation->ele[i] = 1.;
    }
    delete action_tmp;
    
    // Compute the differences in task space and weight them
    getController()->computeDX(dx_des, x_des, activation);
    
    // Compute IK from dx_des
    ActionModelIK::ikFromDX(q_des, q_dot_des, dt);
    
    // Print if debug level is exceeded
    REXEC(7) {
        MatNd_printComment("q_des", q_des);
        MatNd_printComment("q_dot_des", q_dot_des);
        if (T_des) {
            MatNd_printComment("T_des", T_des);
        }
    }
}

void AMIKControllerActivation::getStableAction(MatNd* action) const
{
    // All zero activations is stable
    MatNd_setZero(action);
}

TaskCombinationMethod AMIKControllerActivation::checkTaskCombinationMethod(std::string tcmName)
{
    TaskCombinationMethod tcm;
    if (tcmName == "sum") {
        tcm = TaskCombinationMethod::Sum;
    }
    else if (tcmName == "mean") {
        tcm = TaskCombinationMethod::Mean;
    }
    else if (tcmName == "softmax") {
        tcm = TaskCombinationMethod::SoftMax;
    }
    else if (tcmName == "product") {
        tcm = TaskCombinationMethod::Product;
    }
    else {
        std::ostringstream os;
        os << "Unsupported task combination method: " << tcmName << "! Supported methods: sum, mean, softmax, product.";
        throw std::invalid_argument(os.str());
    }
    return tcm;
}

const char* AMIKControllerActivation::getTaskCombinationMethodName() const
{
    if (taskCombinationMethod == TaskCombinationMethod::Sum) {
        return "sum";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::Mean) {
        return "mean";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::SoftMax) {
        return "softmax";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::Product) {
        return "product";
    }
    else {
        return nullptr;
    }
}

MatNd* AMIKControllerActivation::getActivation() const
{
    return activation;
}

MatNd* AMIKControllerActivation::getXdes() const
{
    return x_des;
}

void AMIKControllerActivation::setXdes(const MatNd* x_des)
{
    // Creating a new x_des and shallow copying saves us from the unexpected size error that clone yields
    this->x_des = MatNd_createLike(x_des);
    MatNd_copy(this->x_des, x_des);
}

void AMIKControllerActivation::setXdesFromTaskSpec(std::vector<PropertySource*>& taskSpec)
{
    MatNd* x_des = MatNd_create(1, 1); // dummy row necessary for MatNd_appendRows
    
    if (taskSpec.size() != getController()->getNumberOfTasks()) {
        std::ostringstream os;
        os << "Received " << taskSpec.size() << " elements in taskSpec, but there are "
           << getController()->getNumberOfTasks() << " Rcs controller tasks!";
        throw std::runtime_error(os.str());
    }
    
    for (PropertySource* ts : taskSpec) {
        MatNd* x_des_temp = nullptr;
        if (!ts->getProperty(x_des_temp, "x_des")) {
            throw std::invalid_argument("Field x_des is missing for at least one task specification!");
        }
        MatNd_appendRows(x_des, x_des_temp);
    }
    MatNd_deleteRows(x_des, 0, 0); // delete dummy row
    this->setXdes(x_des);
    MatNd_destroy(x_des);
}

} /* namespace Rcs */
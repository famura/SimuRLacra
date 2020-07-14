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

#include "OMTask.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>

#include <algorithm>
#include <limits>

namespace Rcs
{

OMTask::OMTask(Task* task) : task(task), maxVelocity(task->getDim(), std::numeric_limits<double>::infinity()) {}

OMTask::~OMTask()
{
    delete task;
}

unsigned int OMTask::getStateDim() const
{
    return task->getDim();
}

void OMTask::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    task->computeX(state);
    task->computeXp(velocity);
}

void OMTask::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    for (size_t i = 0; i < getStateDim(); ++i) {
        minState[i] = task->getParameter(i).minVal;
        maxState[i] = task->getParameter(i).maxVal;
        maxVelocity[i] = this->maxVelocity[i];
    }
}

std::vector<std::string> OMTask::getStateNames() const
{
    // The dynamical systems do report their var names, but unfortunately also include the unit in that string.
    // We have to strip that.
    std::vector<std::string> result;
    result.reserve(getStateDim());
    
    std::string prefix = task->getEffector()->name;
    prefix += "_";
    
    for (auto param : task->getParameters()) {
        auto paramName = param.name;
        auto spaceIdx = paramName.find(' ');
        
        if (spaceIdx != std::string::npos) {
            paramName = paramName.substr(0, spaceIdx);
        }
        
        result.push_back(prefix + paramName);
    }
    return result;
}

OMTask* OMTask::setMinState(double minState)
{
    for (size_t i = 0; i < getStateDim(); ++i) {
        task->getParameter(i).minVal = minState;
    }
    return this;
}

OMTask* OMTask::setMinState(std::vector<double> minState)
{
    RCHECK_EQ(getStateDim(), minState.size());
    for (size_t i = 0; i < getStateDim(); ++i) {
        task->getParameter(i).minVal = minState[i];
    }
    return this;
}

OMTask* OMTask::setMaxState(double maxState)
{
    for (size_t i = 0; i < getStateDim(); ++i) {
        task->getParameter(i).maxVal = maxState;
    }
    return this;
}

OMTask* OMTask::setMaxState(std::vector<double> maxState)
{
    RCHECK_EQ(getStateDim(), maxState.size());
    for (size_t i = 0; i < getStateDim(); ++i) {
        task->getParameter(i).maxVal = maxState[i];
    }
    return this;
}

OMTask* OMTask::setMaxVelocity(double maxVelocity)
{
    std::fill(this->maxVelocity.begin(), this->maxVelocity.end(), maxVelocity);
    return this;
}

OMTask* OMTask::setMaxVelocity(std::vector<double> maxVelocity)
{
    RCHECK_EQ(getStateDim(), maxVelocity.size());
    this->maxVelocity = maxVelocity;
    return this;
}

void OMTask::initTaskBodyNames(const char* effectorName, const char* refBodyName, const char* refFrameName)
{
    if (effectorName != NULL) {
        RcsBody* effector = RcsGraph_getBodyByName(task->getGraph(), effectorName);
        RCHECK_MSG(effector, "Effector body %s not found!", effectorName);
        task->setEffector(effector);
    }
    if (refBodyName != NULL) {
        RcsBody* refBody = RcsGraph_getBodyByName(task->getGraph(), refBodyName);
        RCHECK_MSG(refBody, "Reference body %s not found!", refBodyName);
        task->setRefBody(refBody);
        
        // If there is no separate reference frame, the relative coordinates should be in the frame of the reference body
        if (refFrameName == NULL) {
            task->setRefFrame(refBody);
        }
    }
    if (refFrameName != NULL) {
        RcsBody* refFrame = RcsGraph_getBodyByName(task->getGraph(), refFrameName);
        RCHECK_MSG(refFrame, "Reference frame %s not found!", refFrameName);
        task->setRefFrame(refFrame);
    }
}

} /* namespace Rcs */


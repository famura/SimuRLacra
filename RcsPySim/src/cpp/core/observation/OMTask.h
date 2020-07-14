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

#ifndef _OMTASK_H_
#define _OMTASK_H_

#include "ObservationModel.h"

#include <Task.h>

namespace Rcs
{

/**
 * ObservationModel wrapping a Rcs Task.
 *
 * Note: By default, the state min/max is taken from the task, and the maximum velocity is set to infinity.
 * Use the various setters to change these limits. All limit setters return this for chanining.
 */
class OMTask : public ObservationModel
{
public:
    /**
     * Wrap the given task. Takes ownership of the task object.
     */
    OMTask(Task* task);
    
    virtual ~OMTask();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMTask)
    
    virtual unsigned int getStateDim() const;
    
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
    
    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;
    
    virtual std::vector<std::string> getStateNames() const;
    
    /**
     * Set the lower state limit, broadcasting one value to all elements.
     */
    OMTask* setMinState(double minState);
    
    /**
     * Set the lower state limit. The number of elements must match the state dimension.
     */
    OMTask* setMinState(std::vector<double> minState);
    
    /**
     * Set the upper state limit, broadcasting one value to all elements.
     */
    OMTask* setMaxState(double maxState);
    
    /**
     * Set the upper state limit. The number of elements must match the state dimension.
     */
    OMTask* setMaxState(std::vector<double> maxState);
    
    /**
     * Set the velocity limit, broadcasting one value to all elements.
     */
    OMTask* setMaxVelocity(double maxVelocity);
    
    /**
     * Set the velocity limit. The number of elements must match the state dimension.
     */
    OMTask* setMaxVelocity(std::vector<double> maxVelocity);
    
    /**
     * Return the wrapped Rcs Task.
     */
    Task* getTask() const { return task; }

protected:
    /**
     * Initialize the task's effector, refBody and refFrame values by looking up the named bodies from the graph.
     *
     * @param effectorName Name of effector body, a.k.a. the body controlled by the task.
     * @param refBodyName  Name of reference body, a.k.a. the body the task coordinates
     *                     should be relative to. Set to NULL to use the world origin.
     * @param refFrameName Name of the reference frame body. The task coordinates will
     *                     be expressed in this body's frame if set. If this is NULL,
     *                     refBodyName will be used.
     */
    void initTaskBodyNames(const char* effectorName, const char* refBodyName, const char* refFrameName);

private:
    //! Wrapped task object (owned!)
    Task* task;
    
    //! Settable maximum velocity (min/max state is stored in task parameter)
    std::vector<double> maxVelocity;
};

} /* namespace Rcs */

#endif /* _OMTASK_H_ */

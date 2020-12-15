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

#ifndef _AMIKCONTROLLERACTIVATION_H_
#define _AMIKCONTROLLERACTIVATION_H_

#include <TaskGenericIK.h>
#include "../config/PropertySource.h"
#include "ActionModelIK.h"

namespace Rcs
{

/*! Action model controlling the activations of multiple tasks. Each task is defined by a Rcs IK controller task.
 * For every task, there is one activation variable as part of the action space.
 * The activation is a value between 0 and 1, where 0 means to ignore the task
 * completely. The activation values do not need to sum to 1.
 */
class AMIKControllerActivation : public AMIKGeneric
{
public:
    
    /*! Constructor
     * @param[in] tcm     Mode that determines how the different tasks a.k.a. movement primitives are combined
     */
    explicit AMIKControllerActivation(RcsGraph* graph, TaskCombinationMethod tcm);
    
    virtual ~AMIKControllerActivation();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(AMControllerActivation)
    
    //! Add Rcs controller tasks that are wlways active. Do this after adding the regular tasks.
    void addAlwaysActiveTask(Task* task);
    
    //! Get the number of tasks multiplied by their individual dimension, owned by the action model, except the always active tasks
    virtual unsigned int getDim() const;
    
    virtual void getMinMax(double* min, double* max) const;
    
    virtual std::vector<std::string> getNames() const;
    
    void reset();
    
    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);
    
    virtual void getStableAction(MatNd* action) const;
    
    MatNd* getActivation() const;
    
    MatNd* getXdes() const;
    
    void setXdes(const MatNd* x_des);
    
    void setXdesFromTaskSpec(std::vector<PropertySource*>& taskSpec);
    
    static TaskCombinationMethod checkTaskCombinationMethod(std::string tcmName);
    
    const char* getTaskCombinationMethodName() const;

protected:
    //! The goal in task space
    MatNd* x_des;
    //! Store the cumulative number of task dimensions which are always active
    unsigned int dimAlwaysActiveTasks;
    //! The activation resulting from the action and the task combination method (used for logging)
    MatNd* activation;
    //! Way to combine the tasks' contribution
    TaskCombinationMethod taskCombinationMethod;
};

} /* namespace Rcs */

#endif /* _AMIKCONTROLLERACTIVATION_H_ */

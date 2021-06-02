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

#ifndef _AMDYNAMICALSYSTEMACTIVATION_H_
#define _AMDYNAMICALSYSTEMACTIVATION_H_

#include "ActionModel.h"
#include "DynamicalSystem.h"

namespace Rcs
{

/*! Action model controlling the activations of multiple tasks. Each task is defined by a DynamicalSystem.
 * For every task, there is one activation variable as part of the action space.
 * The activation is a value between 0 and 1, where 0 means to ignore the task
 * completely. The activation values do not need to sum to 1.
 */
class AMDynamicalSystemActivation : public ActionModel
{
public:
    
    /*! Constructor
     * @param[in] wrapped ActionModel defining the output space of the tasks (takes ownership)
     * @param[in] ds      List of dynamical systems (takes ownership)
     * @param[in] tcm     Mode that determines how the different tasks a.k.a. movement primitives are combined
     */
    explicit AMDynamicalSystemActivation(
        ActionModel* wrapped, std::vector<DynamicalSystem*> ds, TaskCombinationMethod tcm);
    
    virtual ~AMDynamicalSystemActivation();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(AMDynamicalSystemActivation)
    
    virtual ActionModel* clone(RcsGraph* newGraph) const;
    
    //! Get the number of DS, i.e. entries in the dynamicalSystems vector, owned by the action model
    virtual unsigned int getDim() const;
    
    virtual void getMinMax(double* min, double* max) const;
    
    virtual std::vector<std::string> getNames() const;
    
    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);
    
    virtual void reset();
    
    virtual void getStableAction(MatNd* action) const;
    
    Eigen::VectorXd getX() const;
    
    Eigen::VectorXd getXdot() const;
    
    virtual ActionModel* getWrappedActionModel() const;
    
    //! Get a vector of the owned dynamical systems.
    const std::vector<DynamicalSystem*>& getDynamicalSystems() const;
    
    MatNd* getActivation() const;
    
    static TaskCombinationMethod checkTaskCombinationMethod(std::string tcmName);
    
    const char* getTaskCombinationMethodName() const;

protected:
    //! Wrapped action model
    ActionModel* wrapped;
    //! List of dynamical systems
    std::vector<DynamicalSystem*> dynamicalSystems;
    //! Current state in task space
    Eigen::VectorXd x;
    //! Current velocity in task space
    Eigen::VectorXd x_dot;
    //! The activation resulting from the action and the task combination method (used for logging)
    MatNd* activation;
    //! Way to combine the tasks' contribution
    TaskCombinationMethod taskCombinationMethod;
};

} /* namespace Rcs */

#endif /* _AMDYNAMICALSYSTEMACTIVATION_H_ */

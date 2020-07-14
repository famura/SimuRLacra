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

#include "AMIntegrate2ndOrder.h"

#include "../util/integrator.h"

#include <Rcs_VecNd.h>
#include <Rcs_macros.h>


namespace Rcs
{

AMIntegrate2ndOrder::AMIntegrate2ndOrder(ActionModel* wrapped, double maxAction) :
    ActionModel(wrapped->getGraph()),
    wrapped(wrapped)
{
    // Create integrator state storage
    integrated_action = MatNd_create(wrapped->getDim(), 1);
    integrated_action_dot = MatNd_create(wrapped->getDim(), 1);
    
    // Store max_action
    this->maxAction = MatNd_create(wrapped->getDim(), 1);
    VecNd_setElementsTo(this->maxAction->ele, maxAction, wrapped->getDim());
}

AMIntegrate2ndOrder::AMIntegrate2ndOrder(ActionModel* wrapped, MatNd* maxAction) :
    ActionModel(wrapped->getGraph()),
    wrapped(wrapped)
{
    // create integrator state storage
    integrated_action = MatNd_create(wrapped->getDim(), 1);
    integrated_action_dot = MatNd_create(wrapped->getDim(), 1);
    
    // store max_action
    RCHECK_MSG(maxAction->m == wrapped->getDim() && maxAction->n == 1, "MaxAction shape must match action dim.");
    this->maxAction = MatNd_clone(maxAction);
}


AMIntegrate2ndOrder::~AMIntegrate2ndOrder()
{
    MatNd_destroy(integrated_action_dot);
    MatNd_destroy(integrated_action);
    MatNd_destroy(maxAction);
    delete wrapped;
}

unsigned int AMIntegrate2ndOrder::getDim() const
{
    return wrapped->getDim();
}

void AMIntegrate2ndOrder::getMinMax(double* min, double* max) const
{
    VecNd_constMul(min, maxAction->ele, -1, getDim());
    VecNd_copy(max, maxAction->ele, getDim());
}

std::vector<std::string> AMIntegrate2ndOrder::getNames() const
{
    auto wnames = wrapped->getNames();
    // add the suffix 'dd' to every var to signal that it's the second order derivative
    for (auto& name : wnames) {
        name += "dd";
    }
    return wnames;
}

void AMIntegrate2ndOrder::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    // perform integration
    intStep2ndOrder(integrated_action, integrated_action_dot, action, dt, IntMode::ForwardEuler);
    
    // pass integrated values to wrapped
    wrapped->computeCommand(q_des, q_dot_des, T_des, integrated_action, dt);
}

void AMIntegrate2ndOrder::reset()
{
    wrapped->reset();
    // reset integrator state to current (initial)
    wrapped->getStableAction(integrated_action);
    MatNd_setZero(integrated_action_dot);
}

void AMIntegrate2ndOrder::getStableAction(MatNd* action) const
{
    // acceleration of 0 is stable
    MatNd_setZero(action);
}

ActionModel* AMIntegrate2ndOrder::getWrappedActionModel() const
{
    return wrapped;
}

ActionModel* AMIntegrate2ndOrder::clone(RcsGraph* newGraph) const
{
    return new AMIntegrate2ndOrder(wrapped->clone(newGraph), maxAction->ele[0]);
}

} /* namespace Rcs */

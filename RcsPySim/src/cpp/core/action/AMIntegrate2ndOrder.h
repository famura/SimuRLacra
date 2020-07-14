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

#ifndef _AMINTEGRATE2NDORDER_H_
#define _AMINTEGRATE2NDORDER_H_

#include "ActionModel.h"

namespace Rcs
{

/**
 * Integrates action values once and passes them to a wrapped action model.
 *
 * This allows to use a position based action model like the inverse kinematics,
 * but command it velocities instead.
 */
class AMIntegrate2ndOrder : public ActionModel
{
private:
    // inner action model, will get the integrated actions
    ActionModel* wrapped;
    // maximum action magnitude, will be used to create the action space
    MatNd* maxAction;
    
    // current values of the integrator
    MatNd* integrated_action;
    MatNd* integrated_action_dot;

public:
    /**
     * Constructor.
     *
     * Takes ownership of the passed inner action model.
     *
     * @param wrapped inner action model
     * @param maxAction maximum action value, reported in the action space.
     */
    AMIntegrate2ndOrder(ActionModel* wrapped, double maxAction);
    
    /**
     * Constructor.
     *
     * Takes ownership of the passed inner action model.
     *
     * @param wrapped inner action model
     * @param maxAction maximum action values, size must match wrapped->getDim().
     *                  Does not take ownership, values are copied.
     */
    AMIntegrate2ndOrder(ActionModel* wrapped, MatNd* maxAction);
    
    virtual ~AMIntegrate2ndOrder();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(AMIntegrate2ndOrder)
    
    virtual ActionModel* clone(RcsGraph* newGraph) const;
    
    virtual unsigned int getDim() const;
    
    virtual void getMinMax(double* min, double* max) const;
    
    virtual std::vector<std::string> getNames() const;
    
    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);
    
    virtual void reset();
    
    virtual void getStableAction(MatNd* action) const;
    
    virtual ActionModel* getWrappedActionModel() const;
};

} /* namespace Rcs */

#endif //_AMINTEGRATE2NDORDER_H_

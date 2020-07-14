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

#ifndef _AMNORMALIZED_H_
#define _AMNORMALIZED_H_

#include "ActionModel.h"

namespace Rcs
{

/**
 * Wraps another action model to accept normalized action values in the range [-1, 1].
 * The passed action values are denormalized and then passed to the wrapped action model.
 */
class AMNormalized : public ActionModel
{
private:
    // Wrapped action model
    ActionModel* wrapped;
    // Scale factor for every action value
    MatNd* scale;
    // Shift for every action value, applied after scale.
    MatNd* shift;

public:
    AMNormalized(ActionModel* wrapped);
    
    virtual ~AMNormalized();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(AMNormalized)
    
    virtual ActionModel* clone(RcsGraph* newGraph) const;
    
    virtual unsigned int getDim() const;
    
    virtual void getMinMax(double* min, double* max) const;
    
    virtual std::vector<std::string> getNames() const;
    
    virtual void computeCommand(
        MatNd* q_des, MatNd* q_dot_des, MatNd* T_des,
        const MatNd* action, double dt);
    
    virtual void reset();
    
    virtual void getStableAction(MatNd* action) const;
    
    virtual ActionModel* getWrappedActionModel() const;
};

} /* namespace Rcs */

#endif /* _AMNORMALIZED_H_ */

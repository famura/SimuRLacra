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

#include "AMNormalized.h"

#include <Rcs_VecNd.h>
#include <Rcs_macros.h>

namespace Rcs
{

AMNormalized::AMNormalized(ActionModel* wrapped) :
    ActionModel(wrapped->getGraph()), wrapped(wrapped)
{
    // Compute scale and shift from inner model bounds
    const MatNd* iModMin = wrapped->getSpace()->getMin();
    const MatNd* iModMax = wrapped->getSpace()->getMax();
    
    // shift is selected so that the median of min and max is 0
    // shift = min + (max - min)/2
    shift = MatNd_clone(iModMax);
    MatNd_subSelf(shift, iModMin);
    MatNd_constMulSelf(shift, 0.5);
    MatNd_addSelf(shift, iModMin);
    
    // scale = (max - min)/2
    scale = MatNd_clone(iModMax);
    MatNd_subSelf(scale, iModMin);
    MatNd_constMulSelf(scale, 0.5);
}

AMNormalized::~AMNormalized()
{
    delete wrapped;
}

unsigned int AMNormalized::getDim() const
{
    return wrapped->getDim();
}

void AMNormalized::getMinMax(double* min, double* max) const
{
    VecNd_setElementsTo(min, -1, getDim());
    VecNd_setElementsTo(max, 1, getDim());
}

std::vector<std::string> AMNormalized::getNames() const
{
    return wrapped->getNames();
}

void AMNormalized::computeCommand(
    MatNd* q_des, MatNd* q_dot_des, MatNd* T_des,
    const MatNd* action, double dt)
{
    // use temp storage for denormalized action, so that action remains unchanged.
    MatNd* denormalizedAction = NULL;
    MatNd_create2(denormalizedAction, action->m, action->n);
    // denormalize: denAction = action * scale + shift
    MatNd_eleMul(denormalizedAction, action, scale);
    MatNd_addSelf(denormalizedAction, shift);
    
    MatNd_maxSelf(denormalizedAction, wrapped->getSpace()->getMin());
    MatNd_minSelf(denormalizedAction, wrapped->getSpace()->getMax());
    
    //MatNd_printTranspose(denormalizedAction);
    
    // call wrapper
    wrapped->computeCommand(q_des, q_dot_des, T_des, denormalizedAction, dt);
    // destroy temp storage
    MatNd_destroy(denormalizedAction);
}

void AMNormalized::reset()
{
    wrapped->reset();
}

void AMNormalized::getStableAction(MatNd* action) const
{
    // compute wrapped stable action
    wrapped->getStableAction(action);
    // and normalize it
    for (unsigned int i = 0; i < getDim(); ++i) {
        action->ele[i] = (action->ele[i] - shift->ele[i])/scale->ele[i];
    }
}

ActionModel* AMNormalized::getWrappedActionModel() const
{
    return wrapped;
}

ActionModel* AMNormalized::clone(RcsGraph* newGraph) const
{
    return new AMNormalized(wrapped->clone(newGraph));
}

} /* namespace Rcs */


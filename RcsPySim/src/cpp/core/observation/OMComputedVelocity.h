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

#ifndef _OMCOMPUTEDVELOCITY_H
#define _OMCOMPUTEDVELOCITY_H

#include "ObservationModel.h"

namespace Rcs
{

/*!
 * An observation model that computes the velocity using finite differences.
 * Use this observation model if the velocity can not be or is not observed directly from the simulation.
 */
class OMComputedVelocity : public ObservationModel
{
public:
    OMComputedVelocity();
    
    virtual ~OMComputedVelocity();
    
    // DO NOT OVERRIDE!
    void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const final;
    
    /**
     * Overridden to initialize lastState.
     * If a subclass wants to override this, make sure to call the base method. Since this method has to call
     * computeState, make sure any relevant initialization is done before.
     */
    virtual void reset();
    
    /**
     * Implement to fill the observation vector with the observed state values. The velocity will be computed automatically.
     * @param[out] state state observation vector to fill, has getStateDim() elements.
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken. Will be 0 if called during reset().
     */
    virtual void computeState(double* state, const MatNd* currentAction, double dt) const = 0;

private:
    // state during last call to computeObservation
    MatNd* lastState;
};

} /* namespace Rcs */

#endif //_OMCOMPUTEDVELOCITY_H

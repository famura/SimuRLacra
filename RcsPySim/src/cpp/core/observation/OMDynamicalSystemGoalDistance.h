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

#ifndef _OMDYNAMICALSYSTEMGOALDISTANCE_H_
#define _OMDYNAMICALSYSTEMGOALDISTANCE_H_

#include "ObservationModel.h"
#include "../action/AMDynamicalSystemActivation.h"


namespace Rcs
{

class ControllerBase;

/**
 * ObservationModel wrapping multiple AMDynamicalSystemActivation to compute the distances to the individuals goals of the
 * dynamical systems but not the rate of change of these goal distances.
 */
class OMDynamicalSystemGoalDistance : public ObservationModel
{
public:
    /**
     * Create from action model.
     * NOTE: assumes that the task activation action model wraps a IK-based action model.
     */
    OMDynamicalSystemGoalDistance(AMDynamicalSystemActivation* actionModel);
    
    virtual ~OMDynamicalSystemGoalDistance();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMDynamicalSystemGoalDistance)
    
    virtual unsigned int getStateDim() const;
    
    virtual unsigned int getVelocityDim() const;
    
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
    
    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;
    
    virtual std::vector<std::string> getStateNames() const;


private:
    //! Task activation action model, provides the tasks (not owned)
    AMDynamicalSystemActivation* actionModel;
    //! Controller to extract the task space state
    ControllerBase* controller;
    //! Limits
    double maxDistance; // by default infinity
};

} /* namespace Rcs */

#endif /* _OMDYNAMICALSYSTEMGOALDISTANCE_H_ */

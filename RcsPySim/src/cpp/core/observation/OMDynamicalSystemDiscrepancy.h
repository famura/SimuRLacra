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

#ifndef _OMDYNAMICALSYSTEMDISCREPANCY_H_
#define _OMDYNAMICALSYSTEMDISCREPANCY_H_

#include "ObservationModel.h"
#include "../action/AMDynamicalSystemActivation.h"


namespace Rcs
{

class ControllerBase;

/**
 * ObservationModel wrapping multiple AMDynamicalSystemActivation to compute the discrepancies between the task space changes
 * commanded by the DS and the ones executed by the robot.
 */
class OMDynamicalSystemDiscrepancy : public ObservationModel
{
public:
    /**
     * Create from action model.
     * NOTE: assumes that the task activation action model wraps a IK-based action model.
     */
    explicit OMDynamicalSystemDiscrepancy(AMDynamicalSystemActivation* actionModel);
    
    virtual ~OMDynamicalSystemDiscrepancy();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(OMDynamicalSystemDiscrepancy)
    
    virtual unsigned int getStateDim() const;
    
    unsigned int getVelocityDim() const override;
    
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
    
    void reset() override;
    
    virtual std::vector<std::string> getStateNames() const;


private:
    // Task activation action model, provides the tasks (not owned)
    AMDynamicalSystemActivation* actionModel;
    
    // Controller to extract the task space state
    ControllerBase* controller;
    
    // last task space state
    MatNd* x_curr;
};

} /* namespace Rcs */

#endif /* _OMDYNAMICALSYSTEMDISCREPANCY_H_ */

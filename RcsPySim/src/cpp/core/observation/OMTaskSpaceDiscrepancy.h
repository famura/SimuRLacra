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

#ifndef _OMTASKSPACEDISCREPANCY_H
#define _OMTASKSPACEDISCREPANCY_H

#include "ObservationModel.h"


namespace Rcs
{

class ControllerBase;

/**
 * ObservationModel computing the discrepancy between a body's position in the desired graph (owned by the controller)
 * and the current graph (owned by the config) in task space.
 */
class OMTaskSpaceDiscrepancy : public ObservationModel
{
public:
    /**
     * Create from action model.
     * NOTE: assumes that the task activation action model wraps a IK-based action model.
     */
    explicit OMTaskSpaceDiscrepancy(
        const char* bodyName,
        const RcsGraph* controllerGraph,
        const RcsGraph* configGraph,
        double maxDiscrepancy = 1.
    );
    
    virtual ~OMTaskSpaceDiscrepancy();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(OMDynamicalSystemDiscrepancy)
    
    virtual unsigned int getStateDim() const;
    
    unsigned int getVelocityDim() const override;
    
    void getLimits(double* minState, double* maxState, double* maxVelocity) const override;
    
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
    
    void reset() override;
    
    virtual std::vector<std::string> getStateNames() const;

private:
    // Body of interest to determine the discrepancy between, e.g. the end-effector
    RcsBody* bodyController;
    RcsBody* bodyConfig;
    
    // We make the design decision that more than maxDiscrepancy difference are not acceptable in any case
    double maxDiscrepancy;  // default is 1 [m, m/s, rad, rad/s]
};

} /* namespace Rcs */

#endif //_OMTASKSPACEDISCREPANCY_H

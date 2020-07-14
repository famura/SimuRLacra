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

#ifndef _OMNORMALIZED_H_
#define _OMNORMALIZED_H_

#include "ObservationModel.h"
#include "../config/PropertySource.h"

namespace Rcs
{

/*!
 * Observation model wrapper to normalize observations into the [-1,1] range.
 */
class OMNormalized : public ObservationModel
{
private:
    // Wrapped observation model
    ObservationModel* wrapped;
    // Scale factor for every observation value. Contains both state and velocity.
    MatNd* scale;
    // Shift for every observation value, applied before scale. Contains both state and velocity.
    MatNd* shift;
public:
    /*!
     * Constructor
     * @param wrapped inner observation model. Takes ownership.
     * @param overrideMin overridden lower bounds by state name
     * @param overrideMax overridden upper bounds by state name
     */
    explicit OMNormalized(ObservationModel* wrapped, PropertySource* overrideMin, PropertySource* overrideMax);
    
    virtual ~OMNormalized();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMNormalized)
    
    virtual unsigned int getStateDim() const;
    
    virtual unsigned int getVelocityDim() const;
    
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
    
    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;
    
    virtual void reset();
    
    virtual std::vector<std::string> getStateNames() const;
    
    virtual std::vector<ObservationModel*> getNested() const;
    
    virtual std::vector<std::string> getVelocityNames() const;
    
};
    
} /* namespace Rcs */

#endif //_OMNORMALIZED_H_

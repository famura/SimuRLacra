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

#ifndef _PHYSICSPARAMETERDESCRIPTOR_H_
#define _PHYSICSPARAMETERDESCRIPTOR_H_

#include "BodyParamInfo.h"
#include "../config/PropertySource.h"
#include "../config/PropertySink.h"

namespace Rcs
{

/**
 * Descriptor for one or more physical parameters settable from Python.
 *
 * The parameters should be stored on the BodyParamInfo reference.
 */
class PhysicsParameterDescriptor
{
protected:
    // body to set parameters on
    BodyParamInfo* bodyParamInfo;
public:
    PhysicsParameterDescriptor();
    
    virtual ~PhysicsParameterDescriptor();
    
    /**
     * Read values from graph and put them into the given dict.
     */
    virtual void getValues(PropertySink* outValues) = 0;
    
    /**
     * Read values from the given dict and apply them to the graph.
     * The parameter names need to be the same as in Rcs, e.g. rolling_friction_coefficient.
     */
    virtual void setValues(PropertySource* inValues) = 0;

protected:
    
    friend class PhysicsParameterManager;
    
    friend class PPDCompound;
    
    /**
     * Setup descriptor to work on the given body reference.
     * Override for more custom initialization.
     */
    virtual void init(BodyParamInfo* bodyParamInfo);
};

} /* namespace Rcs */

#endif /* _PHYSICSPARAMETERDESCRIPTOR_H_ */

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

#ifndef _PPDMATERIALPROPERTIES_H_
#define _PPDMATERIALPROPERTIES_H_

#include "PhysicsParameterDescriptor.h"

namespace Rcs
{

/**
 * Descriptor for the body's friction and rolling friction coefficients as well as other material related properties.
 *
 * Exposes:
 *
 * - friction_coefficient
 *   Linear friction coefficient
 *   Unitless
 * - rolling_friction_coefficient
 *   Linear friction coefficient
 *   Unit: m, multiply unitless coefficient with contact surface curvature.
 *
 * Vortex only (see vortex documentation for details):
 * - slip
 * - compliance
 */
class PPDMaterialProperties : public PhysicsParameterDescriptor
{
public:
    PPDMaterialProperties();
    
    virtual ~PPDMaterialProperties();
    
    
    virtual void getValues(PropertySink* outValues);
    
    virtual void setValues(PropertySource* inValues);
};

} /* namespace Rcs */

#endif /* _PPDMATERIALPROPERTIES_H_ */

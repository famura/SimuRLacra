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

#ifndef _PPDMASSPROPERTIES_H_
#define _PPDMASSPROPERTIES_H_

#include "PPDCompound.h"

namespace Rcs
{

/**
 * Descriptor for the body's mass, center of mass and inertia.
 *
 * If the center of mass or the inertia are not set, they will be calculated automatically.
 * Setting one part of com or inertia is enough to define them as set.
 *
 * Exposes:
 *
 * - mass
 *   Mass of the body
 *   Unit: kg
 * - com_x, com_y, com_z
 *   3d position of the center of mass
 *   Unit: m
 * - i_xx, i_xy, i_xz, i_yy, i_yz, i_zz
 *   Components of the inertia tensor
 *   Unit: kg m^2
 */
class PPDMassProperties : public PPDCompound
{
public:
    PPDMassProperties();
    
    virtual ~PPDMassProperties();
    
    virtual void setValues(PropertySource* inValues);
};

} /* namespace Rcs */

#endif /* _PPDMASSPROPERTIES_H_ */

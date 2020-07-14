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

#ifndef _PPDRODLENGTH_H
#define _PPDRODLENGTH_H

#include "PhysicsParameterDescriptor.h"

namespace Rcs
{

/**
 * Adjusts a rod's / cylinder's length and shift it accordingly.
 * It assumes that the rod is aligned with the body's z axis.
 * This class has be written with the QuanserQube in mind
 * The mass properties are not adjusted automatically, so you should put a PPDMassProperties behind this descriptor.
 */
class PPDRodLength : public PhysicsParameterDescriptor
{
private:
    // Name of value to read
    std::string propertyName;
    // The rod shape
    RcsShape* rodShape;
    
    // Direction of rod, defaults to z
    double rodDirection[3];
    // Offset between rod start and body origin
    double rodOffset[3];
    
    // The child joint at the rod end. This is set for the arm body and used to adjust the pendulum joint position
    RcsJoint* childJoint;
    // Offset between joint origin and rod end, plus rodOffset. Thus the joint pos is jointOffset + rodDir * length.
    double jointOffset[3];

protected:
    virtual void init(BodyParamInfo* bpi);

public:
    PPDRodLength();
    
    virtual void getValues(PropertySink* outValues);
    
    virtual void setValues(PropertySource* inValues);
};

}

#endif //_PPDRODLENGTH_H

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

#ifndef _BODYPARAMINFO_H
#define _BODYPARAMINFO_H

#include <Rcs_graph.h>

#include <PhysicsConfig.h>

#include <string>

namespace Rcs
{

/**
 * Information on a body and changing physics params.
 *
 * Apart from the body, this struct holds material parameters not storable in RcsBody, and flags to track modified parameters.
 */
struct BodyParamInfo
{
    enum ModifiedFlag
    {
        // body mass changed
        MOD_MASS = 1 << 0,
        // body center of gravity changed
        MOD_COM = 1 << 1,
        // body inertia changed
        MOD_INERTIA = 1 << 2,
        // collision shapes changed
        MOD_SHAPE = 1 << 3,
        // position changed
        MOD_POSITION = 1 << 4
    };
    
    // The graph containing the body
    RcsGraph* graph;
    
    // The body
    RcsBody* body;
    
    // prefix for parameter names
    std::string paramNamePrefix;
    
    // body material - this is the material of the body's first shape.
    PhysicsMaterial material;
    
    // flags tracking the modified state of the body
    int modifiedFlag;
    
    BodyParamInfo(RcsGraph* graph, const char* bodyName, PhysicsConfig* physicsConfig);
    
    // reset all change flags
    void resetChanged();
    
    // test if a parameter changed
    inline bool isChanged(int flag)
    {
        return (modifiedFlag & flag) == flag;
    }
    
    
    // mark if a parameter as changed
    inline void markChanged(int flag)
    {
        modifiedFlag |= flag;
    }
};

} // namespace Rcs

#endif //_BODYPARAMINFO_H

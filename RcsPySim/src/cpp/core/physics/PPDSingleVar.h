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

#ifndef _PPDSINGLEVAR_H_
#define _PPDSINGLEVAR_H_

#include "PhysicsParameterDescriptor.h"

#include <functional>

namespace Rcs
{

/**
 * Descriptor for a single scalar variable of type T.
 * The actual parameter name is built to be [lower case body name]_[param name].
 */
template<typename T>
class PPDSingleVar : public PhysicsParameterDescriptor
{
public:
    /*!
     * Returns a reference to the variable whihc this descriptor uses.
     */
    typedef std::function<T & (BodyParamInfo & )> VariableAccessor;
    
    /**
     * Constructor.
     * @param name unprefixed name for the parameter
     * @param modifiedFlag modified flag value to set for BodyParamInfo
     * @param variableAccessor variable accessor function. Returns a reference to be readable and writable.
     */
    PPDSingleVar(std::string name, int modifiedFlag, VariableAccessor variableAccessor) :
        name(std::move(name)), modifiedFlag(modifiedFlag), variableAccessor(variableAccessor) {}
    
    virtual ~PPDSingleVar() = default;
    
    virtual void getValues(PropertySink* outValues)
    {
        outValues->setProperty(name.c_str(), variableAccessor(*bodyParamInfo));
    }
    
    virtual void setValues(PropertySource* inValues)
    {
        // Get value reference
        T& ref = variableAccessor(*bodyParamInfo);
        
        // Try to get from dict
        bool set = inValues->getProperty(ref, name.c_str());
        
        // Set changed flag
        if (set) {
            bodyParamInfo->markChanged(modifiedFlag);
        }
    }

protected:
    virtual void init(BodyParamInfo* bodyParamInfo)
    {
        PhysicsParameterDescriptor::init(bodyParamInfo);
        name = bodyParamInfo->paramNamePrefix + name;
    }

private:
    //! Parameter name/key which the init() method will add a body prefix to this
    std::string name;
    
    //! Value to or to BodyParamInfo::modifiedFlag when modified
    int modifiedFlag;
    
    //! Variable accessor function
    VariableAccessor variableAccessor;
};


} /* namespace Rcs */

#endif /* _PPDSINGLEVAR_H_ */

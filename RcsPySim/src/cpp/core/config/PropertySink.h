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

#ifndef _PROPERTYSINK_H_
#define _PROPERTYSINK_H_

#include "PropertySource.h"

#include <Rcs_MatNd.h>

namespace Rcs
{

/**
 * Mutable version of PropertySource.
 *
 * Note that this class does not specify any mechanism for persisting the changes made here.
 */
class PropertySink : public PropertySource
{
public:
    PropertySink();
    
    virtual ~PropertySink();
    
    /**
     * Set a property to the given string value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, const std::string& value) = 0;
    
    /**
     * Set a property to the given boolean value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, bool value) = 0;
    
    /**
     * Set a property to the given integer value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, int value) = 0;
    
    /**
     * Set a property to the given double value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, double value) = 0;
    
    /**
     * Set a property to the given vector/matrix value.
     *
     * If one of the dimensions of the matrix has size 1,
     * it is interpreted as vector.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, MatNd* value) = 0;
    
    /**
     * Obtain a child property sink.
     *
     * This only adapts the return type from the PropertySource definition.
     */
    virtual PropertySink* getChild(const char* prefix) = 0;
    
    // only adapt the return type
    virtual PropertySink* clone() const = 0;
};

} /* namespace Rcs */

#endif /* _PROPERTYSINK_H_ */

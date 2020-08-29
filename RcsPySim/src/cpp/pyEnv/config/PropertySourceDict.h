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

#ifndef _PROPERTYSOURCEDICT_H_
#define _PROPERTYSOURCEDICT_H_

#include <config/PropertySource.h>
#include <config/PropertySink.h>

#include <pybind11/pybind11.h>

#include <map>

namespace Rcs
{

/**
 * PropertySource/Sink backed by a C++ dict.
 */
class PropertySourceDict : public PropertySink
{
private:
    pybind11::dict dict;
    
    std::map<std::string, PropertySourceDict*> children;
    std::map<std::string, std::vector<PropertySource*>> listChildren;
    
    // for lazy parent writing
    PropertySourceDict* parent;
    const char* prefix;
    bool _exists;
    
    // create child
    PropertySourceDict(
        pybind11::dict dict,
        PropertySourceDict* parent,
        const char* prefix,
        bool exists
    );

public:
    PropertySourceDict(pybind11::dict dict);
    
    virtual ~PropertySourceDict();
    
    virtual bool exists();
    
    virtual bool getProperty(std::string& out, const char* property);

    virtual bool getProperty(std::vector<std::string> &out, const char *property);

    virtual bool getProperty(double& out, const char* property);
    
    virtual bool getProperty(int& out, const char* property);
    
    virtual bool getProperty(MatNd*& out, const char* property);
    
    virtual bool getPropertyBool(const char* property, bool def = false);
    
    virtual PropertySink* getChild(const char* prefix);
    
    // note: not editable for now
    virtual const std::vector<PropertySource*>& getChildList(const char* prefix);
    
    virtual void setProperty(
        const char* property,
        const std::string& value);
    
    virtual void setProperty(const char* property, bool value);
    
    virtual void setProperty(const char* property, int value);
    
    virtual void setProperty(const char* property, double value);
    
    virtual void setProperty(const char* property, MatNd* value);
    
    virtual PropertySink* clone() const;
    
    virtual void saveXML(const char* fileName, const char* rootNodeName);

protected:
    void onWrite();
    
    void appendPrefix(std::ostream&);
};

} /* namespace Rcs */

#endif /* _PROPERTYSOURCEDICT_H_ */

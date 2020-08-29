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

#include "PropertySource.h"
#include "PropertySink.h"

#include <fstream>

namespace Rcs
{

PropertySource::PropertySource()
{
    // nothing to do
}

PropertySource::~PropertySource()
{
    // nothing to do
}


class EmptyPropertySource : public PropertySource
{
public:
    virtual bool exists()
    {
        return false;
    }
    
    virtual bool getProperty(std::string& out, const char* property)
    {
        return false;
    }

    bool getProperty(std::vector<std::string> &out, const char *property) override {
        return false;
    }

    virtual bool getProperty(double& out, const char* property)
    {
        return false;
    }
    
    virtual bool getProperty(int& out, const char* property)
    {
        return false;
    }
    
    virtual bool getProperty(MatNd*& out, const char* property)
    {
        return false;
    }
    
    virtual bool getPropertyBool(const char* property, bool def = false)
    {
        return def;
    }
    
    virtual PropertySource* getChild(const char* prefix)
    {
        return empty();
    }
    
    virtual const std::vector<PropertySource*>& getChildList(const char* prefix)
    {
        static std::vector<PropertySource*> emptyList;
        return emptyList;
    }
    
    virtual PropertySource* clone() const
    {
        return empty();
    }
    
    virtual void saveXML(const char* fileName, const char* rootNodeName)
    {
        std::ofstream out;
        out.open(fileName);
        // no need to use libxml for writing an empty file
        out << "<?xml version=\"1.0\"?>" << std::endl;
        out << "<" << rootNodeName << " />" << std::endl;
    }
};


PropertySource* PropertySource::empty()
{
    static EmptyPropertySource emptySource;
    return &emptySource;
}

// These are just dummies, no extra file needed!

PropertySink::PropertySink()
{
    // Abstract base interface, it's empty!
}

PropertySink::~PropertySink()
{
    // Abstract base interface, it's empty!
}

} /* namespace Rcs */

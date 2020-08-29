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

#include "PropertySourceDict.h"
#include "../util/pybind_dict_utils.h"
#include "../util/type_casters.h"

#include <libxml/tree.h>

#include <algorithm>

namespace Rcs
{

PropertySourceDict::PropertySourceDict(
    pybind11::dict dict,
    PropertySourceDict* parent, const char* prefix, bool exists) :
    dict(dict), parent(parent), prefix(prefix), _exists(exists)
{
    // internal ctor, sets all parts
}

PropertySourceDict::PropertySourceDict(pybind11::dict dict) :
    dict(dict), parent(NULL), prefix(NULL), _exists(true)
{
    // nothing special to do
}

PropertySourceDict::~PropertySourceDict()
{
    // delete children where needed
    for (auto& it: children) {
        if (it.second != empty()) {
            delete it.second;
        }
    }
    for (auto& it: listChildren) {
        for (auto le : it.second) {
            delete le;
        }
    }
}

bool PropertySourceDict::exists()
{
    return _exists;
}

bool PropertySourceDict::getProperty(std::string& out, const char* property)
{
    if (!_exists) {
        return false;
    }
    // try to retrieve
    pybind11::handle elem;
    if (!try_get(dict, property, elem) || elem.is_none()) {
        return false;
    }
    
    // cast to string
    pybind11::str str(elem);
    out = str;
    return true;
}

bool PropertySourceDict::getProperty(std::vector<std::string> &out, const char *property) {
    if (!_exists) {
        return false;
    }
    // try to retrieve
    pybind11::handle elem;
    if (!try_get(dict, property, elem) || elem.is_none()) {
        return false;
    }

    if (py::isinstance<py::str>(elem)) {
        py::str str(elem);
        out.push_back(str);
        return true;
    } else if (py::isinstance<py::iterable>(elem)) {
        for (auto part : elem) {
            py::str str(part);
            out.push_back(str);
        }
        return true;
    } else {
        std::ostringstream os;
        os << "Unsupported value for property entry at ";
        appendPrefix(os);
        os << property << ": Expected string or iterable of string, but got ";
        auto childType = py::str(elem.get_type());
        os << childType.cast<std::string>();
        throw std::invalid_argument(os.str());
    }
}

bool PropertySourceDict::getProperty(double& out, const char* property)
{
    if (!_exists) {
        return false;
    }
    // try to retrieve
    pybind11::handle elem;
    if (!try_get(dict, property, elem) || elem.is_none()) {
        return false;
    }
    
    // cast
    out = elem.cast<double>();
    return true;
}

bool PropertySourceDict::getProperty(int& out, const char* property)
{
    if (!_exists) {
        return false;
    }
    // try to retrieve
    pybind11::handle elem;
    if (!try_get(dict, property, elem) || elem.is_none()) {
        return false;
    }
    
    // cast
    out = elem.cast<int>();
    return true;
}

bool PropertySourceDict::getProperty(MatNd*& out, const char* property)
{
    if (!_exists) {
        return false;
    }
    // try to retrieve
    pybind11::handle elem;
    if (!try_get(dict, property, elem) || elem.is_none()) {
        return false;
    }
    
    // cast. must use internals since handle.cast() doesn't allow pointers
    py::detail::make_caster<MatNd> caster;
    py::detail::load_type(caster, elem);
    out = MatNd_clone(py::detail::cast_op<MatNd*>(caster));
    return true;
}

bool PropertySourceDict::getPropertyBool(const char* property, bool def)
{
    if (!_exists) {
        return def;
    }
    return get_cast<bool>(dict, property, def);
}

PropertySink* PropertySourceDict::getChild(const char* prefix)
{
    std::string prefixStr = prefix;
    // check if it exists already
    auto iter = children.find(prefixStr);
    if (iter != children.end()) {
        return iter->second;
    }
    // try to get child dict
    pybind11::object child;
    bool exists = this->_exists;
    if (exists) {
        exists = try_get(dict, prefix, child) && !child.is_none();
    }
    if (!exists) {
        // not found, use a fresh dict
        child = pybind11::dict();
    }
    else if (!pybind11::isinstance<pybind11::dict>(child)) {
        // exists but not a dict
        std::ostringstream os;
        os << "Unsupported value for child property entry at ";
        appendPrefix(os);
        os << prefix << ": Expected dict, but got ";
        auto childType = pybind11::str(child.get_type());
        os << childType.cast<std::string>();
        throw std::invalid_argument(os.str());
    }
    auto* result = new PropertySourceDict(child, this, prefix, exists);
    children[prefixStr] = result;
    return result;
}

void PropertySourceDict::appendPrefix(std::ostream& os)
{
    if (parent != NULL) {
        parent->appendPrefix(os);
        os << prefix << ".";
    }
}

// most of these are easy
void PropertySourceDict::setProperty(
    const char* property,
    const std::string& value)
{
    dict[property] = value;
    
    onWrite();
}

void PropertySourceDict::setProperty(const char* property, bool value)
{
    dict[property] = value;
    
    onWrite();
}

void PropertySourceDict::setProperty(const char* property, int value)
{
    dict[property] = value;
    
    onWrite();
}

void PropertySourceDict::setProperty(const char* property, double value)
{
    dict[property] = value;
    
    onWrite();
}

void PropertySourceDict::setProperty(const char* property, MatNd* value)
{
    // make sure we use a copy
    dict[property] = pybind11::cast(value, pybind11::return_value_policy::copy);
    
    onWrite();
}

void PropertySourceDict::onWrite()
{
    if (parent != NULL) {
        // notify parent
        parent->onWrite();
        
        
        // put to parent if new
        if (!_exists) {
            parent->dict[prefix] = dict;
        }
    }
}

const std::vector<PropertySource*>& PropertySourceDict::getChildList(const char* prefix)
{
    std::string prefixStr = prefix;
    // Check if it exists already
    auto iter = listChildren.find(prefixStr);
    if (iter != listChildren.end()) {
        return iter->second;
    }
    // Create new entry
    auto& list = listChildren[prefixStr];
    // Retrieve entry from dict
    pybind11::object child;
    bool exists = this->_exists;
    if (exists) {
        exists = try_get(dict, prefix, child) && !child.is_none();
    }
    if (exists) {
        // Parse entry sequence
        if (pybind11::isinstance<pybind11::dict>(child)) {
            // Single element
            list.push_back(new PropertySourceDict(child, this, prefix, true));
        }
        else if (pybind11::isinstance<pybind11::iterable>(child)) {
            // Element sequence
            unsigned int i = 0;
            for (auto elem : child) {
                if (pybind11::isinstance<pybind11::dict>(elem)) {
                    std::ostringstream itemPrefix;
                    itemPrefix << prefix << "[" << i << "]";
                    // Add element
                    list.push_back(new PropertySourceDict(pybind11::reinterpret_borrow<pybind11::dict>(elem), this,
                                                          itemPrefix.str().c_str(), true));
                }
                else {
                    // Exists but not a dict
                    std::ostringstream os;
                    os << "Unsupported element for child property entry at ";
                    appendPrefix(os);
                    os << prefix << "[" << i << "]" << ": Expected dict, but got ";
                    auto childType = pybind11::str(child.get_type());
                    os << childType.cast<std::string>();
                    throw std::invalid_argument(os.str());
                }
                i++;
            }
        }
        else {
            // Exists but not a dict
            std::ostringstream os;
            os << "Unsupported value for child list property entry at ";
            appendPrefix(os);
            os << prefix << ": Expected list of dict or dict, but got ";
            auto childType = pybind11::str(child.get_type());
            os << childType.cast<std::string>();
            throw std::invalid_argument(os.str());
        }
    }
    
    return list;
}

PropertySink* PropertySourceDict::clone() const
{
    // use python deepcopy to copy the dict.
    py::object copymod = py::module::import("copy");
    py::dict cpdict = copymod.attr("deepcopy")(dict);
    
    return new PropertySourceDict(cpdict);
}

static std::string spaceJoinedString(py::handle iterable)
{
    std::ostringstream os;
    
    for (auto ele : iterable) {
        // Append stringified element
        os << py::cast<std::string>(py::str(ele)) << ' ';
    }
    std::string result = os.str();
    // Remove trailing ' '
    if (!result.empty()) {
        result.erase(result.end() - 1);
    }
    return result;
}

static xmlNodePtr dict2xml(const py::dict& data, xmlDocPtr doc, const char* nodeName)
{
    xmlNodePtr node = xmlNewDocNode(doc, NULL, BAD_CAST nodeName, NULL);
    // add children
    for (auto entry : data) {
        auto key = py::cast<std::string>(entry.first);
        auto value = entry.second;
        
        std::string valueStr;
        // check value type
        if (py::isinstance<py::dict>(value)) {
            // as sub node
            xmlNodePtr subNode = dict2xml(py::reinterpret_borrow<py::dict>(value), doc, key.c_str());
            xmlAddChild(node, subNode);
            continue;
        }
        else if (py::isinstance<py::str>(value)) {
            // handle strings before catching them as iterable
            valueStr = py::cast<std::string>(value);
        }
        else if (py::isinstance<py::array>(value)) {
            // handle arrays first before catching them as iterable
            // we don't really have a proper 2d format, just flatten if needed
            
            valueStr = spaceJoinedString(value.attr("flatten")().attr("tolist")());
        }
        else if (py::isinstance<py::iterable>(value)) {
            // an iterable
            
            auto it = value.begin();
            // check if empty
            if (it == value.end()) continue;
            
            // check if child list
            if (std::all_of(it, value.end(), [](py::handle ele) {
                return py::isinstance<py::dict>(ele);
            })) {
                // child list, add one element per child
                for (auto ele : value) {
                    xmlNodePtr subNode = dict2xml(py::reinterpret_borrow<py::dict>(ele), doc, key.c_str());
                    xmlAddChild(node, subNode);
                }
                continue;
            }
            // treat as array-like; a space-joined string
            valueStr = spaceJoinedString(value);
        }
        else {
            // use default str() format for other cases
            valueStr = py::cast<std::string>(py::str(value));
        }
        
        xmlSetProp(node, BAD_CAST key.c_str(), BAD_CAST valueStr.c_str());
    }
    
    return node;
}

void PropertySourceDict::saveXML(const char* fileName, const char* rootNodeName)
{
    // create xml doc
    std::unique_ptr<xmlDoc, void (*)(xmlDocPtr)> doc(xmlNewDoc(NULL), xmlFreeDoc);
    
    // convert root node
    xmlNodePtr rootNode = dict2xml(dict, doc.get(), rootNodeName);
    xmlDocSetRootElement(doc.get(), rootNode);
    
    // perform save
    xmlIndentTreeOutput = 1;
    xmlSaveFormatFile(fileName, doc.get(), 1);
}

} /* namespace Rcs */


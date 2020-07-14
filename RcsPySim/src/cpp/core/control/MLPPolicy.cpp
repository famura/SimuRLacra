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

#include "MLPPolicy.h"

#include <Rcs_parser.h>
#include <Rcs_stlParser.h>
#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>

#include <sstream>
#include <cmath>

namespace Rcs
{

// NONLINEARITIES

void MatNd_tanh_inplace(MatNd* inout)
{
    for (unsigned int i = 0; i < inout->m*inout->n; ++i) {
        inout->ele[i] = std::tanh(inout->ele[i]);
    }
}

void loadParamsFromXml(xmlNodePtr node, std::map<std::string, MatNd*>& parameters)
{
    // lookup nodes for every parameter
    for (auto& entry : parameters) {
        xmlNodePtr paramNode = getXMLChildByName(node, entry.first.c_str());
        
        if (paramNode != NULL) {
            MatNd* paramStore = entry.second;
            // verify parameter shape
            if (paramStore->n == 1) {
                // 1d column array
                unsigned int shape;
                RCHECK(getXMLNodePropertyUnsignedInt(paramNode, "shape", &shape));
                RCHECK_EQ(shape, paramStore->m);
            }
            else if (paramStore->m == 1) {
                // 1d row array
                unsigned int shape;
                RCHECK(getXMLNodePropertyUnsignedInt(paramNode, "shape", &shape));
                RCHECK_EQ(shape, paramStore->n);
            }
            else {
                // 2d array
                unsigned int shape[2];
                RCHECK(getXMLNodePropertyUnsignedIntN(paramNode, "shape", shape, 2));
                RCHECK_EQ(shape[0], paramStore->m);
                RCHECK_EQ(shape[1], paramStore->n);
            }
            
            // load parameter values
            //RCHECK(getXMLNodeVecN(paramNode, paramStore->ele, paramStore->m * paramStore->n));
            // unfortunately, getXMLNodeVecN only supports strings of up to 512 bytes, but a large matrix can be much larger
            // thus we need custom code
            
            // get content from node
            xmlChar* txt = xmlNodeGetContent(paramNode);
            RCHECK(txt);
            // copy content into an std::string
            std::string content((char*) txt);
            // free data from xml
            xmlFree(txt);
            
            // put it into a stringstream
            std::istringstream is(content);
            // be sure to use C locale
            is.imbue(std::locale("C"));
            
            // iterate over elements
            unsigned int nEle = paramStore->m*paramStore->n;
            for (unsigned int i = 0; i < nEle; ++i) {
                // read it from stream
                if (!(is >> paramStore->ele[i])) {
                    RFATAL("Invalid matrix entry at (%d, %d)", i/paramStore->n, i%paramStore->n);
                }
            }
            
            //MatNd_printComment(entry.first.c_str(), paramStore);
        }
    }
}

static MLPPolicy::Nonlinearity loadNonlinearity(xmlNodePtr node, const char* key)
{
    std::string functionName;
    if (!getXMLNodePropertySTLString(node, key, functionName)) {
        return NULL;
    }
    
    // check function name
    if (functionName == "tanh") {
        return MatNd_tanh_inplace;
    }
    else {
        RFATAL("Unsupported nonlinearity '%s' for %s", functionName.c_str(), key);
    }
}

MLPPolicy* loadMLPPolicyFromXmlNode(
    xmlNodePtr node,
    unsigned int input_size,
    unsigned int output_size)
{
    // load policy description
    std::vector<std::string> hidden_size_str;
    getXMLNodePropertyVecSTLString(node, "hidden_sizes", hidden_size_str);
    std::vector<unsigned int> hidden_sizes;
    for (auto& str : hidden_size_str) {
        std::istringstream is(str);
        is.imbue(std::locale("C"));
        
        unsigned int size;
        if (!(is >> size)) {
            // invalid format
            RFATAL("Invalid hidden size entry: %s", str.c_str());
        }
        hidden_sizes.push_back(size);
    }
    
    // load nonlinearities
    MLPPolicy::Nonlinearity hidden_nl = loadNonlinearity(node, "hidden_nonlinearity");
    MLPPolicy::Nonlinearity output_nl = loadNonlinearity(node, "output_nonlinearity");
    
    // create MLP
    MLPPolicy* mlp = new MLPPolicy(input_size, output_size, hidden_sizes, hidden_nl, output_nl);
    
    // load parameters
    auto params = mlp->getParameters();
    loadParamsFromXml(node, params);
    
    return mlp;
}

MLPPolicy* loadMLPPolicyFromXml(const char* xmlFile, unsigned int input_size, unsigned int output_size)
{
    // Determine absolute file name of config file and copy the XML file name
    char filename[256] = "";
    bool fileExists = Rcs_getAbsoluteFileName(xmlFile, filename);
    
    if (!fileExists) {
        RMSG("Resource path is:");
        Rcs_printResourcePath();
        RFATAL("Experiment configuration file \"%s\" not found in "
               "ressource path - exiting", xmlFile ? xmlFile : "NULL");
    }
    
    // load xml tree
    xmlDocPtr doc;
    xmlNodePtr node = parseXMLFile(filename, "MLPPolicy", &doc);
    RCHECK(node);
    
    // create MLP
    MLPPolicy* mlp = loadMLPPolicyFromXmlNode(node, input_size, output_size);
    
    // free xml tree
    xmlFreeDoc(doc);
    
    return mlp;
}

// MLP LAYER

MLPPolicy::MLPLayer::MLPLayer()
{
    // set everything to NULL, will be filled by init
    weights = NULL;
    bias = NULL;
    nonlinearity = NULL;
}

void MLPPolicy::MLPLayer::init(
    unsigned int input_size, unsigned int output_size,
    MLPPolicy::Nonlinearity nonlinearity)
{
    // W = is x os
    weights = MatNd_create(input_size, output_size);
    // b = 1 x os
    bias = MatNd_create(1, output_size);
    
    this->nonlinearity = nonlinearity;
}

MLPPolicy::MLPLayer::~MLPLayer()
{
    MatNd_destroy(weights);
    MatNd_destroy(bias);
}

void MLPPolicy::MLPLayer::compute(MatNd* output, const MatNd* input)
{
    // o = i * W + b
    MatNd_mul(output, input, weights);
    MatNd_addSelf(output, bias);
    // o = NL(o)
    if (nonlinearity != NULL) {
        nonlinearity(output);
    }
}

void MLPPolicy::MLPLayer::getParameters(std::map<std::string, MatNd*>& out, const std::string& prefix)
{
    out[prefix + "W"] = weights;
    out[prefix + "b"] = bias;
}

// HIDDEN LAYER

MLPPolicy::HiddenLayer::HiddenLayer()
{
    // set everything to NULL, will be filled by init
    outputCache = NULL;
}

void MLPPolicy::HiddenLayer::init(
    unsigned int input_size, unsigned int output_size,
    MLPPolicy::Nonlinearity nonlinearity)
{
    MLPLayer::init(input_size, output_size,
                   nonlinearity);
    outputCache = MatNd_create(1, output_size);
}

MLPPolicy::HiddenLayer::~HiddenLayer()
{
    MatNd_destroy(outputCache);
}


// MLP

MLPPolicy::MLPPolicy(
    unsigned int input_size,
    unsigned int output_size,
    std::vector<unsigned int> hidden_sizes,
    MLPPolicy::Nonlinearity hidden_nonlinearity,
    MLPPolicy::Nonlinearity output_nonlinearity)
{
    unsigned int next_layer_input_size = input_size;
    // build hidden layers
    hiddenLayers.resize(hidden_sizes.size());
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        unsigned int layerWidth = hidden_sizes[i];
        // create layer
        hiddenLayers[i].init(next_layer_input_size, layerWidth, hidden_nonlinearity);
        // next layer input size is current layer output size
        next_layer_input_size = layerWidth;
    }
    // build output layer
    outputLayer.init(next_layer_input_size, output_size, output_nonlinearity);
    
    inputSize = input_size;
    outputSize = output_size;
}

MLPPolicy::~MLPPolicy() = default;

void MLPPolicy::computeAction(MatNd* action, const MatNd* observation)
{
    // in rllab, vectors are column vectors. For the MLP formula from rllab, we need them as row vectors
    // luckily, transposing vectors is easy
    RCHECK(observation->n == 1);
    MatNd inputTransposed = MatNd_fromPtr(1, observation->m, observation->ele);
    RCHECK(action->n == 1);
    MatNd outputTransposed = MatNd_fromPtr(1, action->m, action->ele);
    
    // points to the input of the next layer, always non-owning
    const MatNd* nextLayerInput = &inputTransposed;
    // compute hidden layer values
    for (auto& layer : hiddenLayers) {
        layer.compute(layer.outputCache, nextLayerInput);
        // output -> next layer input
        nextLayerInput = layer.outputCache;
    }
    // compute output layer values
    outputLayer.compute(&outputTransposed, nextLayerInput);
}

std::map<std::string, MatNd*> MLPPolicy::getParameters()
{
    std::map<std::string, MatNd*> out;
    
    for (size_t i = 0; i < hiddenLayers.size(); ++i) {
        std::ostringstream prefixBuilder;
        prefixBuilder << "hidden_" << i << ".";
        hiddenLayers[i].getParameters(out, prefixBuilder.str());
    }
    
    outputLayer.getParameters(out, "output.");
    
    return out;
}

// TODO make registerable or delete it

}

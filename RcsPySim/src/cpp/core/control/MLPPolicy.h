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

#ifndef RCSPYSIM_MLPPOLICY_H
#define RCSPYSIM_MLPPOLICY_H

#include "ControlPolicy.h"
#include "../util/BoxSpace.h"

#include <vector>
#include <map>
#include <string>

#include <libxml/tree.h>

namespace Rcs
{

// some useful nonlinearity functions

/**
 * Compute the tanh of every element in place.
 */
void MatNd_tanh_inplace(MatNd* inout);

/**
 * Control policy backed by a multi-layer perceptron.
 *
 * The structure and parameter storage format are designed to be compatible with rllab's GaussianMLPPolicy.
 * Since this is a control policy, only the mean network is used.
 */
class MLPPolicy : public ControlPolicy
{
public:
    typedef void (* Nonlinearity)(MatNd* inout);

private:
    // basic MLP layer structure
    struct MLPLayer
    {
        MatNd* weights;
        MatNd* bias;
        Nonlinearity nonlinearity;
        
        MLPLayer();
        
        void init(unsigned int input_size, unsigned int output_size, Nonlinearity nonlinearity);
        
        ~MLPLayer();
        
        void compute(MatNd* output, const MatNd* input);
        
        void getParameters(std::map<std::string, MatNd*>& out, const std::string& prefix);
    };
    
    // the hidden layers additionally store their output cache matrix
    struct HiddenLayer : public MLPLayer
    {
        MatNd* outputCache;
        
        HiddenLayer();
        
        void init(unsigned int input_size, unsigned int output_size, Nonlinearity nonlinearity);
        
        ~HiddenLayer();
    };
    
    // vector of hidden layers
    std::vector<HiddenLayer> hiddenLayers;
    // the singular output layer
    MLPLayer outputLayer;
    
    unsigned int inputSize, outputSize;
public:
    /**
     * Constructor
     * @param input_size size of the input vector
     * @param output_size size of the output vector
     * @param hidden_sizes list of hidden layer sizes. also determines hidden layer count.
     * @param hidden_nonlinearity nonlinearity to use for hidden layers. Default is tanh.
     * @param output_nonlinearity nonlinearity to use for output layer. Default is NULL a.k.a. none.
     */
    MLPPolicy(
        unsigned int input_size,
        unsigned int output_size,
        std::vector<unsigned int> hidden_sizes,
        Nonlinearity hidden_nonlinearity = MatNd_tanh_inplace,
        Nonlinearity output_nonlinearity = NULL);
    
    virtual ~MLPPolicy();
    
    virtual void computeAction(MatNd* action, const MatNd* observation);
    
    /**
     * Obtain a map from parameter name to parameter storage.
     */
    std::map<std::string, MatNd*> getParameters();
    
    unsigned int getInputSize() const
    {
        return inputSize;
    }
    
    unsigned int getOutputSize() const
    {
        return outputSize;
    }
};

// load parameters from XML
void loadParamsFromXml(xmlNodePtr node, std::map<std::string, MatNd*>& parameters);

MLPPolicy* loadMLPPolicyFromXmlNode(
    xmlNodePtr node,
    unsigned int input_size,
    unsigned int output_size);

MLPPolicy* loadMLPPolicyFromXml(
    const char* xmlFile,
    unsigned int input_size,
    unsigned int output_size);
    
}


#endif //RCSPYSIM_MLPPOLICY_H

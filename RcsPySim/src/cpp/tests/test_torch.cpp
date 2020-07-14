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

#include <catch2/catch.hpp>
#include <torch/torch.h>

#include <iostream>

/**!
 * Inspired by https://pytorch.org/tutorials/advanced/cpp_frontend.html
 */
int createNetAndForward(int64_t numInputs, int64_t numNeurons, int64_t numBatch)
{
    struct Net : torch::nn::Module
    {
        Net(int64_t numInputs, int64_t numNeurons)
        {
            W = register_parameter("W", torch::randn({numInputs, numNeurons}));
            b = register_parameter("b", torch::randn(numNeurons));
        }
        
        torch::Tensor forward(torch::Tensor input)
        {
            return torch::addmm(b, input, W);
        }
        
        torch::Tensor W, b;
    };
    
    // Create the net
    Net net(numInputs, numNeurons);
    
    // Pass one random input
    torch::Tensor inputs = torch::rand({numBatch, numInputs});
    torch::Tensor outputs = net.forward(inputs);
    
    return 0;
}

TEST_CASE("Executing a very basic one layer FNN forward pass", "[PyTorch C++ API]")
{
REQUIRE(createNetAndForward(1, 1, 1)
== 0);
REQUIRE(createNetAndForward(1, 1, 5)
== 0);
REQUIRE(createNetAndForward(1, 3, 1)
== 0);
REQUIRE(createNetAndForward(2, 1, 1)
== 0);
REQUIRE(createNetAndForward(1, 3, 5)
== 0);
REQUIRE(createNetAndForward(2, 3, 1)
== 0);
REQUIRE(createNetAndForward(2, 3, 5)
== 0);
}

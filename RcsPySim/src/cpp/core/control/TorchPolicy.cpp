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

#include "TorchPolicy.h"

#include <Rcs_macros.h>

#include <torch/all.h>

namespace Rcs
{

TorchPolicy::TorchPolicy(const char* filename)
{
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));
    
    // load policy from file
    module = torch::jit::load(filename);
    
    // make sure the module has the right data type
    module.to(torch::kDouble);
}

TorchPolicy::~TorchPolicy()
{
    // nothing to do here
}

void TorchPolicy::reset()
{
    // call a reset method if it exists
    auto resetMethod = module.find_method("reset");
    if (resetMethod.has_value()) {
        torch::jit::Stack stack;
        resetMethod->run(stack);
    }
}

void TorchPolicy::computeAction(MatNd* action, const MatNd* observation)
{
    // assumes observation/action have the proper sizes
    
    // convert input to torch
    torch::Tensor obs_torch = torch::from_blob(
        observation->ele,
        {observation->m},
        torch::dtype(torch::kDouble)
    );
    
    // run it through the module
    torch::Tensor act_torch = module.forward({obs_torch}).toTensor();
    
    // convert output back to rcs matnd.
    torch::Tensor act_out = torch::from_blob(
        action->ele,
        {action->m},
        torch::dtype(torch::kDouble)
    );
    act_out.copy_(act_torch);
}


static ControlPolicyRegistration<TorchPolicy> RegTorchPolicy("torch");

} /* namespace Rcs */

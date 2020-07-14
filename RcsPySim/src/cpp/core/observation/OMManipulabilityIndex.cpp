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

#include <Rcs_typedef.h>
#include "OMManipulabilityIndex.h"

Rcs::OMManipulabilityIndex::OMManipulabilityIndex(Rcs::ActionModelIK* ikModel, bool observeCurrent)
{
    this->controller = new ControllerBase(*ikModel->getController());
    if (observeCurrent) {
        observedGraph = ikModel->getGraph();
    }
    else {
        observedGraph = ikModel->getDesiredGraph();
    }
}

Rcs::OMManipulabilityIndex::~OMManipulabilityIndex()
{
    delete controller;
}

unsigned int Rcs::OMManipulabilityIndex::getStateDim() const
{
    return 1;
}

unsigned int Rcs::OMManipulabilityIndex::getVelocityDim() const
{
    return 0;
}

void Rcs::OMManipulabilityIndex::computeObservation(
    double* state, double* velocity, const MatNd* currentAction,
    double dt) const
{
    RcsGraph_setState(controller->getGraph(), observedGraph->q, observedGraph->q_dot);
    // compute from action model
    state[0] = this->controller->computeManipulabilityCost();
}

std::vector<std::string> Rcs::OMManipulabilityIndex::getStateNames() const
{
    return {"ManipIdx"};
}

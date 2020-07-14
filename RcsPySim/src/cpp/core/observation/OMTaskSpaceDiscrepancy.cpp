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

#include "OMTaskSpaceDiscrepancy.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_Vec3d.h>
#include <Rcs_VecNd.h>


namespace Rcs
{

OMTaskSpaceDiscrepancy::OMTaskSpaceDiscrepancy(
    const char* bodyName,
    const RcsGraph* controllerGraph,
    const RcsGraph* configGraph,
    double maxDiscrepancy
) : maxDiscrepancy(maxDiscrepancy)
{
    bodyController = RcsGraph_getBodyByName(controllerGraph, bodyName);
    bodyConfig = RcsGraph_getBodyByName(configGraph, bodyName);
    RCHECK(bodyController);
    RCHECK(bodyConfig);
}

OMTaskSpaceDiscrepancy::~OMTaskSpaceDiscrepancy()
{
    // Pointer on bodies are destroyed by the graph
}

unsigned int OMTaskSpaceDiscrepancy::getStateDim() const
{
    return 3; // only Cartesian position difference
}

unsigned int OMTaskSpaceDiscrepancy::getVelocityDim() const
{
    return 0;  // does not have a velocity field
}

void OMTaskSpaceDiscrepancy::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    unsigned int sd = getStateDim();
    VecNd_setElementsTo(minState, -maxDiscrepancy, sd);
    VecNd_setElementsTo(maxState, maxDiscrepancy, sd);
    VecNd_setElementsTo(maxVelocity, 0., getVelocityDim());
}

void OMTaskSpaceDiscrepancy::computeObservation(
    double* state,
    double* velocity,
    const MatNd* currentAction,
    double dt) const
{
    // Get the difference (desired - current)
    Vec3d_sub(state, bodyController->A_BI->org, bodyConfig->A_BI->org);
    
    // Print if debug level is exceeded
    REXEC(7) {
        std::cout << "Task space discrepancy: " << state << std::endl;
    }
}

void OMTaskSpaceDiscrepancy::reset()
{
//      RcsBody* bodyController = RcsGraph_getBodyByName(controllerGraph, bodyName);
//      RcsBody* bodyConfig = RcsGraph_getBodyByName(configGraph, bodyName);
}

std::vector<std::string> OMTaskSpaceDiscrepancy::getStateNames() const
{
    std::vector<std::string> result;
    result.reserve(getStateDim());
    result.push_back(std::string(bodyConfig->name) + "_DiscrepTS_X");
    result.push_back(std::string(bodyConfig->name) + "_DiscrepTS_Y");
    result.push_back(std::string(bodyConfig->name) + "_DiscrepTS_Z");
    return result;
}

} /* namespace Rcs */
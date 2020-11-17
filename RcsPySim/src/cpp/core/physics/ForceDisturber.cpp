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

#include "ForceDisturber.h"

#include <Rcs_typedef.h>
#include <Rcs_Vec3d.h>

namespace Rcs
{

ForceDisturber::ForceDisturber(RcsBody* body, RcsBody* refFrame) : body(body), refFrame(refFrame)
{
    Vec3d_setZero(lastForce);
}

ForceDisturber::~ForceDisturber()
{
    // Nothing here to destroy
}

void ForceDisturber::apply(Rcs::PhysicsBase* sim, double force[3])
{
    // this is somewhat a bug in Rcs: The body map uses the bodies from the simulator's internal graph.
    // so, we need to convert this
    RcsBody* simBody = RcsGraph_getBodyByName(sim->getGraph(), body->name);
    
    // Transform force if needed
    double forceLocal[3];
    if (refFrame != nullptr) {
        Vec3d_rotate(forceLocal, refFrame->A_BI->rot, force);
        Vec3d_transRotateSelf(forceLocal, body->A_BI->rot);
    }
    else {
        Vec3d_copy(forceLocal, force);
    }
    
    // Store for UI
    Vec3d_copy(lastForce, force);
    
    // Apply the force in the physics simulation
    sim->setForce(simBody, force, NULL);
}

const double* ForceDisturber::getLastForce() const
{
    return lastForce;
}

} /* namespace Rcs */

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>
#include <GraphNode.h>

namespace Rcs
{

void ForceDisturber::addToViewer(GraphNode* graphNode)
{
    // Obtain graph node (assuming there's only one)
//    BodyNode* bn = graphNode->getBodyNode(body);
    // TODO add to viewer (required to draw arrows)
}

} /* namespace Rcs */

#else


void Rcs::ForceDisturber::addToViewer(GraphNode* graphNode)
{
    // nop
}

#endif

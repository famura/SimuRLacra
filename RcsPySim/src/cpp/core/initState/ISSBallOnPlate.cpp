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

#include "ISSBallOnPlate.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

#include <algorithm>    // std::min


namespace Rcs
{

ISSBallOnPlate::ISSBallOnPlate(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    ball = RcsGraph_getBodyByName(graph, "Ball");
    RCHECK(ball);
    plate = RcsGraph_getBodyByName(graph, "Plate");
    RCHECK(plate);
    
    // Ensure that the ball's position is either absolute or relative to the plate
    RCHECK_MSG(ball->parent == NULL || ball->parent == plate,
               "The ball's parent must be NULL or the Plate, but was %s",
               ball->parent ? ball->parent->name : "NULL");
    
    // Find plate dimensions
    RcsShape* plateShape = NULL;
    RCSBODY_TRAVERSE_SHAPES(plate) {
        if (SHAPE->type == RCSSHAPE_BOX) {
            // Found the shape
            plateShape = SHAPE;
            break;
        }
    }
    RCHECK_MSG(plateShape != NULL, "Plate body must have a box shape.");
    plateWidth = plateShape->extents[0];
    plateHeight = plateShape->extents[1];
}

ISSBallOnPlate::~ISSBallOnPlate()
{
    // Nothing to destroy
}

unsigned int ISSBallOnPlate::getDim() const
{
    return 2;
}

void ISSBallOnPlate::getMinMax(double* min, double* max) const
{
    // Use a safety margin between the edge of the plate and the ball
    double smallerExtent = std::min(plateWidth/2, plateHeight/2);
    double minDist = smallerExtent*0.6;
    double maxDist = smallerExtent*0.8;
    
    // Set minimum and maximum relative to the plate's center
    min[0] = minDist;
    max[0] = maxDist;
    min[1] = -M_PI;
    max[1] = +M_PI;
}


std::vector<std::string> ISSBallOnPlate::getNames() const
{
    return {"r", "phi"};
}

void ISSBallOnPlate::applyInitialState(const MatNd* initialState)
{
    double ballX = std::cos(initialState->ele[1]) * initialState->ele[0];
    double ballY = std::sin(initialState->ele[1]) * initialState->ele[0];
    
    if (ball->parent == NULL) {
        // The initial position is relative to the plate, so shift it if the ball's rbj are absolute.
        ballX += plate->A_BI->org[0];
        ballY += plate->A_BI->org[1];
    }
    
    // Set the position to the ball's rigid body joints
    double* ballRBJ = &graph->q->ele[ball->jnt->jointIndex];
    ballRBJ[0] = ballX;
    ballRBJ[1] = ballY;
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */

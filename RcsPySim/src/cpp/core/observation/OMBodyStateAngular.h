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

#ifndef _OMBODYSTATEANGULAR_H_
#define _OMBODYSTATEANGULAR_H_

#include "OMTask.h"
#include "OMTaskPositions.h"

namespace Rcs
{

/**
 * Observation model of angular body state.
 * Observes the rotation of the body around all three axis as well as the angular velocity.
 */
class OMBodyStateAngular : public OMTask
{
public:
    /**
     * Constructor
     *
     * @param graph        World to observe
     * @param effectorName Name of effector body, a.k.a. the body controlled by the task
     * @param refBodyName  Name of reference body, a.k.a. the body the task coordinates should be relative to.
     *                     Set to NULL to use the world origin.
     * @param refFrameName Name of the reference frame body. The task coordinates will be expressed in this body's
     *                     frame if set. If this is NULL, refBodyName will be used.
     */
    OMBodyStateAngular(
        RcsGraph* graph,
        const char* effectorName,
        const char* refBodyName = NULL,
        const char* refFrameName = NULL
    );
};

/**
 * Observation model of angular body state.
 * Observes the rotation of the body around all three axis, but not the angular velocity.
 */
class OMBodyStateAngularPositions : public OMTaskPositions
{
public:
    /**
     * Constructor
     *
     * @param graph        World to observe
     * @param effectorName Name of effector body, a.k.a. the body controlled by the task
     * @param refBodyName  Name of reference body, a.k.a. the body the task coordinates should be relative to.
     *                     Set to NULL to use the world origin.
     * @param refFrameName Name of the reference frame body. The task coordinates will be expressed in this body's
     *                     frame if set. If this is NULL, refBodyName will be used.
     */
    OMBodyStateAngularPositions(
        RcsGraph* graph,
        const char* effectorName,
        const char* refBodyName = NULL,
        const char* refFrameName = NULL
    );
};

} /* namespace Rcs */

#endif /* _OMBODYSTATEANGULAR_H_ */

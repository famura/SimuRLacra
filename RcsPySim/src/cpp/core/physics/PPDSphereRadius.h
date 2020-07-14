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

#ifndef _PPDSPHERERADIUS_H_
#define _PPDSPHERERADIUS_H_

#include "PPDSingleVar.h"

namespace Rcs
{

/**
 * Descriptor for the radius of a sphere-shaped body.
 *
 * The sphere must be the first shape of the body for this to work.
 * This is specific to the ball-on-plate task since it also has to adjust
 * the ball's position to prevent it from clipping through the plate.
 *
 * Note that this does not update the inertia based on the shape changes,
 * for that, add PPDMassProperties after this descriptor.
 *
 * Exposes:
 *
 * - radius:
 *   Radius of the sphere [m]
 */
class PPDSphereRadius : public PPDSingleVar<double>
{
public:
    /**
     * Constructor
     *
     * @param shapeIdx The spheres's index within given the body.
     *                 This is given by the order of the shapes in the config xml-file.
     * @param prevBodyName Name of the previous body if the graph to which the sphere is placed relative to,
     *                     Use "" if the sphere is defined in world coordinates
     */
    PPDSphereRadius(std::string prevBodyName, unsigned int shapeIdx = 0, unsigned int shapeIdxPrevBody = 0);
    
    virtual ~PPDSphereRadius();
    
    virtual void setValues(PropertySource* inValues);

protected:
    virtual void init(BodyParamInfo* bodyParamInfo);

private:
    //! Name of the previous body if the graph to which the sphere is placed relative to
    std::string prevBodyName;
    
    //! The spheres's index within given the body. This is given by the order of the shapes in the config xml-file.
    unsigned int shapeIdx;
    //! The spheres's index within the previous body. This is given by the order of the shapes in the config xml-file.
    unsigned int shapeIdxPrevBody;
};

} /* namespace Rcs */

#endif /* _PPDSPHERERADIUS_H_ */

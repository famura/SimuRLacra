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

#ifndef _ISSPLANAR3LINK_H_
#define _ISSPLANAR3LINK_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the planar-3-link environment.
 * The initial state consists of the 3 joint angles.
 */
class ISSPlanar3Link : public InitStateSetter
{
public:
    /**
     * Constructor.
     * The passed graph must contain three bodies named 'Link1', 'Link2', and 'Link3'.
     * @param graph graph to set the state on
     */
    ISSPlanar3Link(RcsGraph* graph);
    
    virtual ~ISSPlanar3Link();
    
    unsigned int getDim() const override;
    
    void getMinMax(double* min, double* max) const override;
    
    virtual std::vector<std::string> getNames() const;
    
    void applyInitialState(const MatNd* initialState) override;

private:
    RcsBody* link1;
    RcsBody* link2;
    RcsBody* link3;
};
    
} /* namespace Rcs */

#endif /* _ISSPLANAR3LINK_H_ */

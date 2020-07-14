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

#ifndef _RCSPYSIM_BOXSPACEPROVIDER_H_
#define _RCSPYSIM_BOXSPACEPROVIDER_H_

#include "BoxSpace.h"
#include "nocopy.h"

namespace Rcs
{

/**
 * A class that lazily provides an 1D box space.
 */
class BoxSpaceProvider
{
private:
    mutable BoxSpace* space;

public:
    BoxSpaceProvider();
    
    virtual ~BoxSpaceProvider();
    
    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(BoxSpaceProvider)
    
    /**
     * Compute and return the space.
     */
    const BoxSpace* getSpace() const;
    
    /**
     * Provides the number of elements in the space.
     * Since the BoxSpace object will be cached, this must not change.
     *
     * @return number of elements for the space.
     */
    virtual unsigned int getDim() const = 0;
    
    /**
     * Provides minimum and maximum values for the space.
     *
     * The passed arrays will be large enough to hold getDim() values.
     *
     * @param[out] min minimum value storage
     * @param[out] max maximum value storage
     */
    virtual void getMinMax(double* min, double* max) const = 0;
    
    /**
     * Provides names for each entry of the space.
     *
     * These are intended for use in python, i.e., for pandas dataframe column names.
     *
     * @return a vector of name strings. Must be of length getDim() or empty.
     */
    virtual std::vector<std::string> getNames() const;
};

} // namespace Rcs


#endif //_RCSPYSIM_BOXSPACEPROVIDER_H_

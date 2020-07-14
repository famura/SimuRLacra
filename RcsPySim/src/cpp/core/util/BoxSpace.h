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

#ifndef _BOXSPACE_H_
#define _BOXSPACE_H_

#include <Rcs_MatNd.h>

#include <string>
#include <vector>

namespace Rcs
{

/**
 * A box in R^n.
 *
 * Each coordinate is bounded by a minimum and maximum value.
 *
 * This class doesn't define any advanced operations, it solely exists to clean up the C++ interface of the implementation.
 */
class BoxSpace
{
public:
    /**
     * Create a box space from min/max value matrices. The shape of the matrices must match.
     *
     * The BoxSpace object will assume ownership of the MatNds.
     *
     * @param min minimum values for each element.
     * @param max maximum values for each element.
     * @param names variable names for each element.
     */
    BoxSpace(MatNd* min, MatNd* max, std::vector<std::string> names = {});
    
    /**
     * Create a box space with the same min/max value for all dimensions.
     *
     * @param min minimum value for any element
     * @param max maximum value for any element
     * @param m number of rows in space
     * @param n number of columns in space
     * @param names variable names for each element.
     */
    BoxSpace(double min, double max, unsigned int m, unsigned int n = 1, std::vector<std::string> names = {});
    
    // copyable
    BoxSpace(const BoxSpace&);
    
    BoxSpace& operator=(const BoxSpace&);
    
    // moveable
    BoxSpace(BoxSpace&&) noexcept;
    
    BoxSpace& operator=(BoxSpace&&) noexcept;
    
    // cleans up owned MatNds
    ~BoxSpace();
    
    
    /**
     * Check if the given MatNd has the right shape for this space.
     *
     * The optional parameter can take an error message to report back.
     *
     * @param[in] values value matrix to check
     * @param[out] msg string variable to write a message to if the values do not fit.
     *
     * @return true if the values fit
     */
    bool checkDimension(const MatNd* values, std::string* msg = NULL) const;
    
    /**
     * Check if the given MatNd has the right shape and it's values are valid.
     *
     * The optional parameter can take an error message to report back.
     *
     * @param[in] values value matrix to check
     * @param[out] msg string variable to write a message to if the values do not fit.
     *
     * @return true if the values fit
     */
    bool contains(const MatNd* values, std::string* msg = NULL) const;
    
    
    /**
     * Create a matrix of the right shape to fit into the space.
     * @return a MatNd of the right shape
     */
    MatNd* createValueMatrix() const;
    
    /**
     * Fill the given matrix out with random values fitting into the space bounds.
     *
     * @param[out] out MatNd to fill. The shape must match.
     */
    void sample(MatNd* out) const;
    
    /**
     * Lower bounds for each variable.
     */
    const MatNd* getMin() const
    {
        return min;
    }
    
    /**
     * Upper bounds for each variable.
     */
    const MatNd* getMax() const
    {
        return max;
    }
    
    /**
     * Names for each variable.
     *
     * These are intended for use in python, i.e., for pandas dataframe column names.
     */
    const std::vector<std::string>& getNames() const
    {
        return names;
    }

private:
    // space bounds
    MatNd* min;
    MatNd* max;
    
    // names for every space entry
    // flattened, row-major
    std::vector<std::string> names;
};

} /* namespace Rcs */

#endif /* _BOXSPACE_H_ */

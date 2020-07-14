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

#include "BoxSpace.h"

#include <Rcs_macros.h>
#include <Rcs_basicMath.h>

#include <sstream>

namespace Rcs
{

static void initDefaultNames(std::vector<std::string>& out, unsigned int dim)
{
    // generate numbered values
    const char* format = "value %02d";
    const size_t BUFFER_LEN = 10;
    char buffer[BUFFER_LEN];
    
    for (unsigned int i = 0; i < dim; ++i) {
        snprintf(buffer, BUFFER_LEN, format, i);
        out.emplace_back(buffer);
    }
}

BoxSpace::BoxSpace(MatNd* min, MatNd* max, std::vector<std::string> names) :
    min(min), max(max), names(names)
{
    RCHECK(min);
    RCHECK(max);
    // validate that the dimensions match
    RCHECK_MSG(min->m == max->m && min->n == max->n,
               "Min (%d, %d) and max (%d, %d) dimensions must match", min->m,
               max->m, min->n, max->n);
    
    // init names with default if any
    if (names.empty()) {
        initDefaultNames(names, min->m*min->n);
    }
    else {
        RCHECK_MSG(min->m*min->n == names.size(), "Name variables must match element count");
    }
}

BoxSpace::BoxSpace(double min, double max, unsigned int m, unsigned int n, std::vector<std::string> names) : names(
    names)
{
    // create matrices according to size
    this->min = MatNd_create(m, n);
    this->max = MatNd_create(m, n);
    
    // fill them with the passed values
    MatNd_setElementsTo(this->min, min);
    MatNd_setElementsTo(this->max, max);
    
    // init names with default if any
    if (names.empty()) {
        initDefaultNames(names, m*n);
    }
    else {
        RCHECK_MSG(m*n == names.size(), "Name variables must match element count");
    }
}

BoxSpace::BoxSpace(const BoxSpace& other)
{
    // copy new
    min = MatNd_clone(other.min);
    max = MatNd_clone(other.max);
}

BoxSpace& BoxSpace::operator=(const BoxSpace& other)
{
    // destroy old
    MatNd_destroy(min);
    MatNd_destroy(max);
    
    // copy new
    min = MatNd_clone(other.min);
    max = MatNd_clone(other.max);
    
    return *this;
}

BoxSpace::BoxSpace(BoxSpace&& other) noexcept
{
    // steal values
    min = other.min;
    max = other.max;
    
    // and set them to NULL over there. The destructor can deal with it.
    other.min = NULL;
    other.max = NULL;
}

BoxSpace& BoxSpace::operator=(BoxSpace&& other) noexcept
{
    // destroy old
    MatNd_destroy(min);
    MatNd_destroy(max);
    
    // steal values
    min = other.min;
    max = other.max;
    
    // and set them to NULL over there. The destructor can deal with it.
    other.min = NULL;
    other.max = NULL;
    
    return *this;
}

BoxSpace::~BoxSpace()
{
    MatNd_destroy(min);
    MatNd_destroy(max);
}

bool BoxSpace::checkDimension(const MatNd* values, std::string* msg) const
{
    if (values->m != min->m || values->n != min->n) {
        // They don't match
        if (msg) {
            std::ostringstream os;
            os << "mismatching dimensions: expected (" << min->m << ", "
               << min->n << ") but got (" << values->m << ", " << values->n
               << ")";
            *msg = os.str();
        }
        return false;
    }
    return true;
}

bool BoxSpace::contains(const MatNd* values, std::string* msg) const
{
    bool valid = true;

    // Check if the dimensions match
    if (!checkDimension(values, msg)) {
        valid = false;
    }

    // Check individual value bounds
    for (unsigned int i = 0; i < values->m*values->n; ++i) {
        bool less = values->ele[i] < min->ele[i];
        bool more = values->ele[i] > max->ele[i];
        if (less || more) {
            // They don't match
            if (msg) {
                std::ostringstream os;
                os << "value out of bounds: val(" << (i/values->n) << ", "
                   << (i%values->n) << ") = " << values->ele[i];
                if (less) {
                    os << " < " << min->ele[i];
                }
                else {
                    os << " > " << max->ele[i];
                }
                *msg = os.str();
            }
            valid = false;
        }
    }
    
    return valid;
}

MatNd* BoxSpace::createValueMatrix() const
{
    return MatNd_create(min->m, min->n);
}

void BoxSpace::sample(MatNd* out) const
{
    RCHECK(out->m == min->m && out->n == min->n);
    // iterate elements
    for (unsigned int i = 0; i < out->m*out->n; ++i) {
        out->ele[i] = Math_getRandomNumber(min->ele[i], max->ele[i]);
    }
}

} /* namespace Rcs */

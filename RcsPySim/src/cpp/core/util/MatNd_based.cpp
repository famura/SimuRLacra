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

#include "MatNd_based.h"

#include <Rcs_macros.h>
#include <Rcs_MatNd.h>

#include <algorithm>
#include <vector>


void MatNd_checkShapeEq(MatNd* m1, MatNd* m2)
{
    RCHECK_MSG((m1->m == m2->m) && (m1->n == m2->n),
               "Shape mismatch between two MatNds: %d x %d != %d x %d",
               m1->m, m1->n, m2->m, m2->n);
}


void findAllBitCombinations(MatNd* allComb)
{
    // Get the dimension
    size_t N = allComb->n;
    
    // Set all entries to 0. The first row will be used directly as a result
    MatNd_setZero(allComb);
    
    size_t i = 1;  // row index
    for (size_t k = 1; k <= N; k++) {
        std::vector<size_t> vec(N, 0);
        std::vector<size_t> currOnes(k, 1);
        // Set last k bits to 1
        std::copy(std::begin(currOnes), std::end(currOnes), std::end(vec) - k);
        
        // Get the combinations / permutations and set them into the matrix
        do {
            for (size_t v = 0; v < vec.size(); v++) {
                // Fill the current row
                MatNd_set2(allComb, i, v, vec[v]);
            }
            i++;
        } while (std::next_permutation(vec.begin(), vec.end()));  // returns false if no valid permutation was found
    }
}

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

#ifndef _INTEGRATOR_H_
#define _INTEGRATOR_H_

#include <Rcs_MatNd.h>

namespace Rcs
{

/**
 * Integration mode.
 */
enum class IntMode
{
    ForwardEuler, // a.k.a. explicit Euler
    BackwardEuler, // a.k.a. implicit Euler
    SymplecticEuler
};

/*-----------------------------------------------------------------------------*
 * Notation
 *-----------------------------------------------------------------------------*/
// 0 indicates old/current time (begin of the integration interval)
// 1, 2, ..., T-1 indicate intermediate time steps
// T indicates the final time (end of the integration interval)

/**
 * Second order integration function.
 *
 * @param[in,out] x current value matrix, updated with new value.
 * @param[in,out] xd current first derivative value matrix, updated with new value.
 * @param[in] xdd second derivative matrix to integrate
 * @param dt timestep length in seconds
 * @param mode ForwardEuler or SymplecticEuler
 */
void intStep2ndOrder(
    MatNd* x, MatNd* xd, const MatNd* xdd,
    double dt, IntMode mode);

/**
 * First order integration function.
 *
 * @param[in,out] x current value matrix, updated with new value.
 * @param[in] xd_0 first derivative matrix at current timestep to integrate. Used by ForwardEuler.
 * @param[in] xd_T first derivative matrix at next timestep to integrate. Used by BackwardEuler.
 * @param dt timestep length in seconds
 * @param mode ForwardEuler or BackwardEuler
 */
void intStep1stOrder(
    MatNd* x, const MatNd* xd_0, const MatNd* xd_T,
    double dt, IntMode mode);
    
} // namespace Rcs

#endif // _INTEGRATOR_H_

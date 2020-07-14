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

#include "integrator.h"

#include <Rcs_macros.h>

namespace Rcs
{

void intStep2ndOrder(MatNd* x, MatNd* xd, const MatNd* xdd, double dt, IntMode mode)
{
    MatNd* aux_d = NULL, * aux_dd = NULL, * aux_sum = NULL;
    MatNd_fromStack(aux_d, x->m, 1);
    MatNd_fromStack(aux_dd, x->m, 1);
    MatNd_fromStack(aux_sum, x->m, 1);
    switch (mode) {
        case IntMode::ForwardEuler: // no energy loss noticed
            MatNd_constMul(aux_dd, xdd, 0.5*dt*dt); // 0.5*xdd_0*dt^2
            MatNd_constMul(aux_d, xd, dt); // xd_0*dt
            MatNd_add(aux_sum, aux_d, aux_dd); // xd_0*dt + 0.5*xdd_0*dt^2
            MatNd_addSelf(x, aux_sum); // x_T = x_0 + dx_0*dt + 0.5*xdd_0^2*dt^2
            
            MatNd_constMulAndAddSelf(xd, xdd, dt); // xd_T = xd_0 + xdd_0*dt
            break;
        case IntMode::SymplecticEuler: // slight energy loss noticed
            
            MatNd_constMulAndAddSelf(xd, xdd, dt); // xd_T = xd_0 + xdd_0*dt
            
            MatNd_constMul(aux_sum, xd, dt); // xd_T*dt
            MatNd_addSelf(x, aux_sum); // x_T = x_0 + x_T*dt
            break;
        default:
            RFATAL("Invalid parameter value 'mode'!");
    }
}

void intStep1stOrder(MatNd* x, const MatNd* xd_0, const MatNd* xd_T, double dt, IntMode mode)
{
    switch (mode) {
        case IntMode::ForwardEuler:
            // x_T = x_0 + xd_0*dt
            MatNd_constMulAndAddSelf(x, xd_0, dt); // x_0 <- x_T (reuse of the variable)
            break;
        case IntMode::BackwardEuler:
            // x_T = x_0 + xd_T*dt
            MatNd_constMulAndAddSelf(x, xd_T, dt); // x_0 <- x_T (reuse of the variable)
            break;
        default:
            RFATAL("Invalid parameter value 'mode'!");
    }
}

}
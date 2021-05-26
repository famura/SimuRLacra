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

#ifndef _ACTIONMODELIKPOLICY_H_
#define _ACTIONMODELIKPOLICY_H_

#include "../action/ActionModelIK.h"
#include "ControlPolicy.h"

namespace Rcs
{

/*!
 * ControlPolicy backed by an ActionModelIK. Takes the ownership of the action model.
 */
class ActionModelIKPolicy : public ControlPolicy
{
public:
    ActionModelIKPolicy(AMIKGeneric* actionModel, double dt);
    
    virtual ~ActionModelIKPolicy();
    
    virtual void reset();
    
    virtual void computeAction(MatNd* action, const MatNd* observation);
    
    void setBotInternals(const MatNd* q_ctrl, const MatNd* qd_ctrl, const MatNd* T_ctrl);

private:
    AMIKGeneric* actionModel;
    
    // Variables from the outside, i.e. PyBot
    MatNd* q_ctrl;
    MatNd* qd_ctrl;
    MatNd* T_ctrl;
    double dt;
};

} /* namespace Rcs */

#endif /* _ACTIONMODELIKPOLICY_H_ */

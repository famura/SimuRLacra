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

#ifndef _RCSPYBOT_H_
#define _RCSPYBOT_H_

#include "ExperimentConfig.h"
#include "DataLogger.h"

#include <MotionControlLayer.h>

#include <mutex>

namespace Rcs
{

class ActionModel;

class ObservationModel;

class ControlPolicy;

// TODO besserer Name
class RcsPyBot : public MotionControlLayer
{
public:
    /**
     * Create the bot from the given property source.
     * @param propertySource configuration
     */
    explicit RcsPyBot(PropertySource* propertySource);
    
    virtual ~RcsPyBot();
    
    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(RcsPyBot)
    
    ExperimentConfig* getConfig()
    {
        return config;
    }
    
    /**
     * Replace the control policy.
     * This method may be called while the bot is running.
     * Setting NULL causes the bot to return to it's initial position.
     * Does NOT take ownership.
     */
    void setControlPolicy(ControlPolicy* controlPolicy);
    
    ControlPolicy* getControlPolicy() const
    {
        std::unique_lock<std::mutex> lock(controlPolicyMutex);
        return controlPolicy;
    }
    
    //! Data logger
    DataLogger logger;
    
    /**
     * Get storage matrix for current observation.
     *
     * WARNING: the contents may update asynchronously. The dimensions are constant.
     */
    MatNd* getObservation() const;
    
    /**
     * Get storage matrix for current action.
     *
     * WARNING: the contents may update asynchronously. The dimensions are constant.
     */
    MatNd* getAction() const;

protected:
    virtual void updateControl();
    
    //! Control policy mutex (mutable to allow using it from const functions)
    mutable std::mutex controlPolicyMutex;
    
    //! Experiment configuration
    ExperimentConfig* config;
    bool allJointsPosCtrl;
    
    //! Control policy
    ControlPolicy* controlPolicy;
    
    // temporary matrices
    MatNd* q_ctrl;
    MatNd* qd_ctrl;
    MatNd* T_ctrl;
    
    MatNd* observation;
    MatNd* action;
};

} /* namespace Rcs */

#endif /* _RCSPYBOT_H_ */

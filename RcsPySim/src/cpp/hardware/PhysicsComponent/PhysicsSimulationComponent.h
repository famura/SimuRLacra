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

#ifndef _PHYSICSSIMULATIONCOMPONENT_H_
#define _PHYSICSSIMULATIONCOMPONENT_H_

#include "HardwareComponent.h"

#include <PhysicsBase.h>
#include <PeriodicCallback.h>


namespace Rcs
{
class PhysicsSimulationComponent : public HardwareComponent, public PeriodicCallback
{
public:
    
    PhysicsSimulationComponent(
        RcsGraph* graph, const char* engine = "Bullet",
        const char* physicsCfgFile = NULL);
    
    PhysicsSimulationComponent(PhysicsBase* sim);
    
    virtual ~PhysicsSimulationComponent();
    
    virtual void updateGraph(RcsGraph* graph);
    
    virtual void setCommand(
        const MatNd* q_des, const MatNd* qp_des,
        const MatNd* T_des);
    
    virtual void tare();
    
    virtual void setEnablePPS(bool enable);
    
    virtual const char* getName() const;
    
    virtual double getCallbackUpdatePeriod() const;
    
    virtual double getLastUpdateTime() const;
    
    virtual void getLastPositionCommand(MatNd* q_des) const;
    
    virtual void setFeedForward(bool ffwd);
    
    virtual PhysicsBase* getPhysicsSimulation() const;
    
    virtual int sprint(char* str, size_t size) const;
    
    virtual void start(double updateFreq = 10.0, int prio = 50);
    
    virtual void setMutex(pthread_mutex_t* mtx);
    
    virtual double getStartTime() const;
    
    bool startThread();
    
    bool stopThread();

private:
    virtual void callback();
    
    void lock();
    
    void unlock();
    
    RcsGraph* currentGraph;
    double dt, dtSim, tStart;
    PhysicsBase* sim;
    pthread_mutex_t* mtx;
    bool ffwd;
};

}

#endif   // _PHYSICSSIMULATIONCOMPONENT_H_

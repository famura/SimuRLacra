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

#ifndef _OMJOINTSTATE_H_
#define _OMJOINTSTATE_H_

#include "ObservationModel.h"

namespace Rcs
{

/*!
 * Observes joint positions for a single joint.
 */
class OMJointState : public ObservationModel
{
public:
    
    static ObservationModel* observeAllJoints(RcsGraph* graph);
    
    static ObservationModel* observeUnconstrainedJoints(RcsGraph* graph);
    
    /*!
     * Constructor
     * @param graph graph to observe
     * @param jointName name of joint to observe
     * @param wrapJointAngle whether to wrap the state of a rotational joint into the [-pi, pi] range.
     *                       Use for unlimited rotation joints.
     */
    OMJointState(RcsGraph* graph, const char* jointName, bool wrapJointAngle);
    
    /*!
     * Constructor
     * Decides to wrap the joint angle if the joint's movement range is exactly [-pi, pi].
     * @param graph graph to observe
     * @param jointName name of joint to observe
     */
    OMJointState(RcsGraph* graph, const char* jointName);
    
    virtual ~OMJointState();
    
    virtual unsigned int getStateDim() const;
    
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
    
    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;
    
    virtual std::vector<std::string> getStateNames() const;

private:
    // create from joint
    OMJointState(RcsGraph* graph, RcsJoint* joint);
    
    // The graph being observed
    RcsGraph* graph;
    // The joint to observe
    RcsJoint* joint;
    // Set to true in order to wrap the joint angle into [-pi, pi].
    bool wrapJointAngle;
};

} /* namespace Rcs */

#endif /* _OMJOINTSTATE_H_ */

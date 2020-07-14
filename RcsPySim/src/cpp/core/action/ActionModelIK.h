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

#ifndef _ACTIONMODELIK_H_
#define _ACTIONMODELIK_H_

#include "ActionModel.h"

#include <ControllerBase.h>
#include <IkSolverRMR.h>

namespace Rcs
{

/*! Base class for IK-based action models.
 *
 * This class encapsulates a Rcs ControllerBase object as well as an inverse kinematics solver.
 *
 * To use this, create a subclass and add your tasks in the constructor. Then, use computeIK(...) to invoke the solver.
 *
 * In order to facilitate stable kinematics, the joint space error returned by the IK solver is integrated into the
 * desired joint states ignoring the current measurements. To faciliate this, the ControllerBase object contains
 * a copy of the main RcsGraph. Thus, all IK tasks should use this graph.
 */
class ActionModelIK : public ActionModel
{

public:
    /*! Create an empty IK-based action model. Tasks must be added to the controller before the first reset() call.
     * @param graph current state graph
     */
    explicit ActionModelIK(RcsGraph* graph);
    
    /*! Create an IK based action model using a fixed set of tasks.
     * @param graph current state graph
     * @param tasks controller tasks to use
     */
    explicit ActionModelIK(RcsGraph* graph, std::vector<Task*> tasks);
    
    virtual ~ActionModelIK();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(ActionModelIK)
    
    virtual void reset();
    
    /*! Exposes a const reference to the controller object.
     * This is readonly since modifying the controller could break the solver.
     */
    const ControllerBase* getController() const { return controller; }
    
    //! The graph holding the desired state of the IK solver.
    RcsGraph* getDesiredGraph() const;
    
    /*!
     * Setup collision avoidance in the IK.
     *
     * Will use a copy of the given collision model which operates on the desired graph.
     * @param modelToCopy collision model to copy
     */
    void setupCollisionModel(const RcsCollisionMdl* modelToCopy);

protected:
    /*! Add a task to the controller.
     * This works exactly as getController()->addTask() would. However, tasks may only be added before the first
     * reset() call. Thus, getController() returns a const reference, and this method makes sure reset has not been
     * called yet.
     * @param[in] task task to add. Takes ownership.
     */
    void addTask(Task* task);
    
    /*! Compute IK solution from desired task space state.
     *
     * @param[out] q_des     resulting desired joint positions
     * @param[out] q_dot_des resulting desired joint velocities
     * @param[out] T_des     resulting desired joint torques
     * @param[in]  x_des     desired task space state
     * @param[in]  dt        time step since the last call
     */
    void computeIK(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* x_des, double dt);
    
    /*! Compute IK solution from desired task space velocity.
     *
     * @param[out] q_des     resulting desired joint positions
     * @param[out] q_dot_des resulting desired joint velocities
     * @param[out] T_des     resulting desired joint torques
     * @param[in]  x_dot_des desired task space velocity
     * @param[in]  dt        time step since the last call
     */
    void computeIKVel(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* x_dot_des, double dt);
    
    // Desired state graph (owned by the controller)
    RcsGraph* desiredGraph;
    
    // Collision model used for collision gradient. Can't put it into the controller because that one has no setter.
    RcsCollisionMdl* collisionMdl;
    
    // shared implementation of computeIK and computeIKVel. Takes input from dx_des.
    void ikFromDX(MatNd* q_des, MatNd* q_dot_des, double dt) const;
    
    // Temporary data storage for the IK
    MatNd* dx_des;
    MatNd* dH;
    MatNd* dq_ref;

private:
    // We friend AMIKGeneric so it can check whether the solver has been initialized
    friend class AMIKGeneric;
    
    //! Holder for the IK tasks (owned)
    ControllerBase* controller;
    //! IK solver (owned)
    IkSolverRMR* solver;
    //! Scaling factor for the nullspace gradient
    double alpha = 1e-4;
    //! Regularization factor for the task space weighting
    double lambda = 1e-6;
};

/*! Generic IK-based action model.
 *
 * Here, the action is used directly as desired task space state for the IK.
 * You need to add your tasks to the controller before using it:
 *
 *     AMIKGeneric* am = new AMIKGeneric(graph);
 *     am->addTask(new TaskXY(...));
 *     am->reset(); // now it can be used, and no new tasks must be added.
 *
 */
class AMIKGeneric : public ActionModelIK
{
public:
    using ActionModelIK::ActionModelIK;
    // Expose addTask
    using ActionModelIK::addTask;
    
    virtual ActionModel* clone(RcsGraph* newGraph) const;
    
    virtual unsigned int getDim() const;
    
    virtual void getMinMax(double* min, double* max) const;
    
    virtual std::vector<std::string> getNames() const;
    
    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);
    
    virtual void getStableAction(MatNd* action) const;
};

} /* namespace Rcs */

#endif /* _ACTIONMODELIK_H_ */

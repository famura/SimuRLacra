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

#ifndef _CONTROLPOLICY_H_
#define _CONTROLPOLICY_H_

#include <Rcs_MatNd.h>

#include <vector>
#include <string>

namespace Rcs
{

class PropertySource;

/**
 * Base class for a control policy that computes the actions from a given observation vector.
 */
class ControlPolicy
{
public:
    // static registry
    
    //! Policy factory function. Should read a policy from a file.
    typedef ControlPolicy* (* ControlPolicyCreateFunction)(const char*);
    
    /*! Register a control policy type.
     * @param[in] name policy type name
     * @param[in] creator factory function
     */
    static void registerType(const char* name, ControlPolicyCreateFunction creator);
    
    /*! Load a saved policy.
     * @param[in] name policy type name
     * @param[in] dataFile file to load
     * @return loaded policy
     */
    static ControlPolicy* create(const char* name, const char* dataFile);
    
    /*! Load a saved policy defined by the given configuration.
     * @param[in] config property config, containing type and file entries
     * @return loaded policy
     */
    static ControlPolicy* create(PropertySource* config);
    
    //! List available policy names.
    static std::vector<std::string> getTypeNames();
    
    ControlPolicy();
    
    virtual ~ControlPolicy();
    
    /*! Reset internal state if any.
     * The default implementation does nothing.
     */
    virtual void reset();
    
    /*!
     * Compute the action according to the policy.
     * @param[out] action matrix to store the action in
     * @param[in]  observation current observed state
     */
    virtual void computeAction(MatNd* action, const MatNd* observation) = 0;
    
    /*!
     * Propagate the robot's internal state to the policy. The default implementation does nothing.
     * @param[in]
     * @param[in] TODO
     * @param[in] TODO
     */
    virtual void setBotInternals(const MatNd* q_ctrl, const MatNd* qd_ctrl, const MatNd* T_ctrl){}
};

/**
 * Create a static field of this type to register a control policy type.
 */
template<class Policy>
class ControlPolicyRegistration
{
public:
    /**
     * Register the template type under the given name.
     * @param name experiment name
     */
    explicit ControlPolicyRegistration(const char* name)
    {
        ControlPolicy::registerType(name, ControlPolicyRegistration::create);
    }
    
    /**
     * Creator function passed to ExperimentConfig::registerType.
     */
    static ControlPolicy* create(const char* dataFile)
    {
        return new Policy(dataFile);
    }
    
};

} /* namespace Rcs */

#endif /* _CONTROLPOLICY_H_ */

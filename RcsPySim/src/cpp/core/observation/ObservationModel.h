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

#ifndef _OBSERVATIONMODEL_H_
#define _OBSERVATIONMODEL_H_

#include "../util/BoxSpaceProvider.h"

#include <Rcs_MatNd.h>
#include <Rcs_graph.h>

namespace Rcs
{

/**
 * The ObservationModel encapsulates the computation of the state vector from the current graph state.
 */
class ObservationModel : public BoxSpaceProvider
{
public:
    
    virtual ~ObservationModel();
    
    /**
     * Create a MatNd for the observation vector, fill it using computeObservation and return it.
     * The caller must take ownership of the returned matrix.
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken
     * @return a new observation vector
     */
    MatNd* computeObservation(const MatNd* currentAction, double dt) const;
    
    /**
     * Fill the given matrix with observation data.
     * @param[out] observation observation output vector
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken
     */
    void computeObservation(MatNd* observation, const MatNd* currentAction, double dt) const;
    
    /**
     * The number of state variables.
     */
    virtual unsigned int getStateDim() const = 0;
    
    /**
     * The number of velocity variables.
     * The default implementation assumes that for each state there is a velocity.
     */
    virtual unsigned int getVelocityDim() const;
    
    /**
     * Implement to fill the observation vector with the observed values.
     * @param[out] state state observation vector to fill, has getStateDim() elements.
     * @param[out] velocity velocity observation vector to fill, has getVelocityDim() elements.
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken
     */
    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const = 0;
    
    /**
     * Provides the minimum and maximum observable values.
     * Since the velocity is symmetric, only the maximum needs to be provided.
     * The default implementation uses -inf and inf.
     * @param[out] minState minimum state vector to fill, has getStateDim() elements.
     * @param[out] maxState maximum state vector to fill, has getStateDim() elements.
     * @param[out] maxVelocity maximum velocity vector to fill, has getVelocityDim() elements.
     */
    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;
    
    /**
     * Reset any internal state. This is called to begin a new episode.
     * It should also reset values depending on modifiable physics parameters.
     * This is an optional operation, so the default implementation does nothing.
     */
    virtual void reset();
    
    /**
     * Provides names for each state entry.
     * @return a vector of name strings. Must be of length getStateDim() or empty for a nameless space.
     */
    virtual std::vector<std::string> getStateNames() const;
    
    /**
     * Provides names for each velocity entry.
     * The default implementation derives the names from getStateNames(), appending a 'd' to each name.
     *
     * @return a vector of name strings. Must be of length getVelocityDim() or empty for a nameless space.
     */
    virtual std::vector<std::string> getVelocityNames() const;
    
    // These functions should not be overridden in subclasses!
    /**
     * Provides the minimum and maximum observable values.
     * Delegates to getLimits.
     */
    virtual void getMinMax(double* min, double* max) const final;
    
    /**
     * The number of observed variables is twice the number of state variables.
     * Delegates to getStateDim.
     */
    virtual unsigned int getDim() const final;
    
    /**
     * The velocity names are the state names postfixed with 'd'.
     * Delegates to getStateNames.
     */
    virtual std::vector<std::string> getNames() const final;
    
    /**
     * Find a nested observation model of a specified type.
     * If multiple observation models match, the first found in depth-first search order is returned.
     * @tparam OM observation model type to find
     * @return nested observation model or NULL if not found.
     */
    template<typename OM>
    OM* findModel()
    {
        auto dc = dynamic_cast<OM*>(this);
        if (dc) {
            return dc;
        }
        for (auto nested : getNested()) {
            dc = nested->findModel<OM>();
            if (dc) {
                return dc;
            }
        }
        return NULL;
    }
    
    //! result of findOffsets
    struct Offsets
    {
        int pos;
        int vel;
        
        operator bool() const
        {
            return pos >= 0;
        }
    };
    
    /**
     * Find a nested observation model of a specified type.
     * If multiple observation models match, the first found in depth-first search order is returned.
     * NOTE: the positions and velovities are done separately. In order to correct for masked state observations use
     *       `observationModel->getStateDim() + thisOM.vel + i` to get the index.
     * @tparam OM observation model type to find
     * @return nested observation model or NULL if not found.
     */
    template<typename OM>
    Offsets findOffsets()
    {
        Offsets local = {0, 0};
        auto dc = dynamic_cast<OM*>(this);
        if (dc) {
            return local;
        }
        for (auto nested : getNested()) {
            auto no = nested->findOffsets<OM>();
            if (no) {
                return {local.pos + no.pos, local.vel + no.vel};
            }
            
            local.pos += nested->getStateDim();
            local.vel += nested->getVelocityDim();
        }
        return {-1, -1};
    }
    
    /**
     * List directly nested observation models.
     * The default implementation returns an empty list, since there are no nested models.
     */
    virtual std::vector<ObservationModel*> getNested() const;
};

} /* namespace Rcs */

#endif /* _OBSERVATIONMODEL_H_ */

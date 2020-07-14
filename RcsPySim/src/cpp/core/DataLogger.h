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

#ifndef SRC_CPP_CORE_DATALOGGER_H_
#define SRC_CPP_CORE_DATALOGGER_H_

#include "util/BoxSpace.h"

#include <Rcs_MatNd.h>

#include <string>
#include <mutex>
#include <fstream>

namespace Rcs
{

/**
 * Logs experiment data to a csv file.
 *
 * For every timestep, it records observations, actions and the reward.
 */
class DataLogger
{
public:
    
    /**
     * Constructor.
     *
     * @param fileBaseName base name for files, will append `filenum`.csv on start
     */
    DataLogger(std::string fileBaseName = "rollout_data_");
    
    virtual ~DataLogger();
    
    /**
     * Start logging for at most stepCount steps.
     *
     * @param[in] observationSpace environment's observation space
     * @param[in] actionSpace environment's action space
     * @param[in] maxStepCount maximum number of time steps
     * @param[in] filename filename to override default generated filename
     */
    void start(
        const BoxSpace* observationSpace, const BoxSpace* actionSpace, unsigned int maxStepCount,
        const char* filename = NULL);
    
    /**
     * Stop logging and flush data to file.
     */
    void stop();
    
    /**
     * Record data for the current step.
     */
    void record(const MatNd* observation, const MatNd* action, double reward);
    
    /**
     * Return true if running.
     */
    bool isRunning() const
    {
        return running;
    }

private:
    // update mutex
    std::recursive_mutex mutex;
    
    // auto log file naming
    std::string baseFileName;
    unsigned int fileCounter;
    
    // true if running
    volatile bool running;
    
    // buffer, avoids writing on realtime main thread
    MatNd* buffer;
    // step counter in current logging run
    unsigned int currentStep;
    
    // current output stream
    std::ofstream output;
};

} /* namespace Rcs */

#endif /* SRC_CPP_CORE_DATALOGGER_H_ */

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

#include "DataLogger.h"

#include <Rcs_macros.h>
#include <Rcs_VecNd.h>

#include <ctime>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace Rcs
{

DataLogger::DataLogger() :
    fileCounter(0),
    running(false),
    buffer(nullptr),
    currentStep(0)
{
    // Nothing else to do
}

DataLogger::~DataLogger()
{
    // Make sure we stopped
    stop();
}

void DataLogger::start(const BoxSpace* observationSpace, const BoxSpace* actionSpace, unsigned int maxStepCount)
{
    // Guard against concurrency
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (running) {
        RLOG(1, "Already running!");
        return;
    }
    running = true;
    // Determine filename
    std::string fname;
    std::time_t t = std::time(nullptr); // get time now
    std::tm* now = std::localtime(&t);
    std::ostringstream os;
    os << (now->tm_year + 1900) << '-' << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-' <<
       std::setfill('0') << std::setw(2) << now->tm_mday << '_' << (fileCounter++) << ".csv";
    fname = os.str();
    
    // Open output file
    output.open(fname);
    
    // Write header (column names)
    output << R"("steps)";
    for (auto& name : observationSpace->getNames()) {
        output << "\",\"" << name;
    }
    for (auto& name : actionSpace->getNames()) {
        output << "\",\"" << name;
    }
    output << "\"" << std::endl;
    
    // Allocate buffer. Use maxStepCount + 1 to also count the final observation
    buffer = MatNd_create(maxStepCount + 1, observationSpace->getNames().size() + actionSpace->getNames().size());
    currentStep = 0;
    
    std::cout << "Logging started!" << std::endl;;
}

void DataLogger::stop()
{
    // Guard against concurrency
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (!running) {
        return;
    }
    running = false;
    
    // Write buffer contents to csv
    for (unsigned int row = 0; row <= currentStep; ++row) { // <= to also write the last row
        // Write step number
        output << row;
        
        // Write elements
        for (unsigned int col = 0; col < buffer->n; ++col) {
            output << "," << MatNd_get2(buffer, row, col);
        }
        output << std::endl;
    }
    
    // Close file
    output.flush();
    output.close();
    output.clear();
    
    // Delete buffer
    MatNd_destroy(buffer);
    
    std::cout << "Logging stopped!" << std::endl;;
}

void DataLogger::record(const MatNd* observation, const MatNd* action)
{
    // Try to obtain lock. If it's blocked, it's blocked by start or stop, so we don't want to lock anyways.
    std::unique_lock<std::recursive_mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock() || !running) {
        return;
    }
    
    // Check if the logging is over. The very last step is index by m
    if (currentStep >= buffer->m - 1) {
        // Add final observations and dummy actions (to later filter them out) to the buffer
        double* lineBuffer = MatNd_getRowPtr(buffer, currentStep);
        VecNd_copy(&lineBuffer[0], observation->ele, observation->m);
        double nanAction[action->m];
        for (unsigned int i = 0; i < action->m; i++) {
            nanAction[i] = NAN;
        }
        VecNd_copy(&lineBuffer[observation->m], nanAction, action->m);
        
        stop();
        return;
    }
    
    // Add a line to the buffer
    double* lineBuffer = MatNd_getRowPtr(buffer, currentStep++);
    VecNd_copy(&lineBuffer[0], observation->ele, observation->m);
    VecNd_copy(&lineBuffer[observation->m], action->ele, action->m);
}

} /* namespace Rcs */

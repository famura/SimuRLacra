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

#include "vortex_log.h"

#include <Rcs_macros.h>
#include <stdexcept>
#include <sstream>

#ifdef USE_VORTEX
// implement method
#include <Vx/VxMessage.h>

void Rcs::setVortexLogLevel(const char* levelStr)
{
    // convert string to enum value
    Vx::eLogLevel level;
    if (STRCASEEQ(levelStr, "off")) {
        level = Vx::kOff;
    }
    else if (STRCASEEQ(levelStr, "fatal")) {
        level = Vx::kFatal;
    }
    else if (STRCASEEQ(levelStr, "error")) {
        level = Vx::kError;
    }
    else if (STRCASEEQ(levelStr, "warn")) {
        level = Vx::kWarn;
    }
    else if (STRCASEEQ(levelStr, "info")) {
        level = Vx::kInfo;
    }
    else if (STRCASEEQ(levelStr, "debug")) {
        level = Vx::kDebug;
    }
    else if (STRCASEEQ(levelStr, "trace")) {
        level = Vx::kTrace;
    }
    else if (STRCASEEQ(levelStr, "all")) {
        level = Vx::kAll;
    }
    else {
        std::ostringstream os;
        os << "Unsupported vortex log level: " << levelStr;
        throw std::invalid_argument(os.str());
    }
    // set to vortex
    Vx::LogSetLevel(level);
}

#else
// vortex not available, show warning
void Rcs::setVortexLogLevel(const char* levelStr)
{
    RLOG(1, "Vortex physics engine is not supported.");
}
#endif




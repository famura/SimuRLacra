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

#include <catch2/catch.hpp>

#include <config/PropertySourceXml.h>
#include <RcsSimEnv.h>

#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>

using namespace Rcs;

TEST_CASE("Environment run")
{
    // Set Rcs debug level
    RcsLogLevel = 2;
    
    // Make sure the resource path is set up
    Rcs_addResourcePath("config");
    
    std::vector<std::string> configs{"config/BallOnPlate/exBotKuka.xml", "config/TargetTracking/exTargetTracking.xml"};
    
    for (auto& configFile : configs) {
        DYNAMIC_SECTION("Config " << configFile) {
            RcsSimEnv env(new PropertySourceXml(configFile.c_str()));
            
            // Reset env
            MatNd* obs = env.reset(PropertySource::empty(), NULL);
            
            // Verify observation
            REQUIRE(env.observationSpace()->checkDimension(obs));
            MatNd_destroy(obs);
            
            MatNd* action = env.actionSpace()->createValueMatrix();
            
            // Perform random steps
            for (int step = 0; step < 100; ++step) {
                // Make a random action
                env.actionSpace()->sample(action);
                
                // Perform step
                obs = env.step(action);
                
                // Cannot really verify observation, an observation outside the space is valid and leads to termination.
                MatNd_destroy(obs);
            }
            
            MatNd_destroy(action);
        }
    }
}

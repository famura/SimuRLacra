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

#include <RcsPyBot.h>
#include <action/ActionModel.h>
#include <config/PropertySourceXml.h>
#include <control/MLPPolicy.h>
#include <observation/ObservationModel.h>
#include <physics/PhysicsParameterManager.h>

#include <Rcs_macros.h>
#include <Rcs_utils.h>
#include <Rcs_timer.h>
#include <Rcs_parser.h>
#include <Rcs_resourcePath.h>

#include <KeyCatcherBase.h>
#include <SegFaultHandler.h>
#include <PhysicsSimulationComponent.h>
#include <Rcs_cmdLine.h>

#ifdef GRAPHICS_AVAILABLE

#include <ViewerComponent.h>
#include <HUD.h>

#endif


RCS_INSTALL_SEGFAULTHANDLER

bool runLoop = true;

/******************************************************************************
 * Ctrl-C destructor. Tries to quit gracefully with the first Ctrl-C
 * press, then just exits.
 *****************************************************************************/
static void quit(int /*sig*/)
{
    static int kHit = 0;
    runLoop = false;
    fprintf(stderr, "Trying to exit gracefully - %dst attempt\n", kHit + 1);
    kHit++;
    
    if (kHit == 2) {
        fprintf(stderr, "Exiting without cleanup\n");
        exit(0);
    }
}

int main(int argc, char** argv)
{
    RMSG("Starting Rcs...");
    
    Rcs::KeyCatcherBase::registerKey("q", "Quit");
    Rcs::KeyCatcherBase::registerKey("l", "Start/stop data logging");
    Rcs::KeyCatcherBase::registerKey("p", "Activate control policy");
    Rcs::KeyCatcherBase::registerKey("o", "Deactivate policy and return to initial state");
    
    // Ctrl-C callback handler
    signal(SIGINT, quit);
    
    // This initialize the xml library and check potential mismatches between the version it was compiled for and
    // the actual shared library used.
    LIBXML_TEST_VERSION;
    
    // Parse command line arguments
    Rcs::CmdLineParser argP(argc, argv);
    char xmlFileName[128] = "ex_p3l-ika_export.xml";
    char directory[128] = "../config/ENVIRONMENT_FOLDER";
    argP.getArgument("-dl", &RcsLogLevel, "Debug level (default is 0)");
    argP.getArgument("-f", xmlFileName, "Configuration file name");
    argP.getArgument("-dir", directory, "Configuration file directory");
    bool valgrind = argP.hasArgument("-valgrind", "Start without GUIs and graphics");
//    bool simpleGraphics = argP.hasArgument("-simpleGraphics", "OpenGL without fancy stuff (shadows, anti-aliasing)");
    
    const char* hgr = getenv("SIT");
    if (hgr != NULL) {
        std::string meshDir = std::string(hgr) + std::string("/Data/RobotMeshes/1.0/data");
        Rcs_addResourcePath(meshDir.c_str());
    }

//    Rcs_addResourcePath("../config");
    Rcs_addResourcePath(directory);
    
    // show help if requested
    if (argP.hasArgument("-h", "Show help message")) {
        Rcs::KeyCatcherBase::printRegisteredKeys();
        Rcs::CmdLineParser::print();
        Rcs_printResourcePath();
        return 0;
    }
    
    // Create simulated robot from config file
    RMSG("Creating robot...");
    Rcs::RcsPyBot bot(new Rcs::PropertySourceXml(xmlFileName));
    
    // TODO add hardware components (which I don't have currently)
    // Add physics simulator for testing
    Rcs::PhysicsParameterManager* ppmanager = bot.getConfig()->createPhysicsParameterManager();
    Rcs::PhysicsBase* simImpl = ppmanager->createSimulator(bot.getConfig()->properties->getChild("initDomainParam"));
    
    bot.getConfig()->actionModel->reset();
    bot.getConfig()->observationModel->reset();
    
    Rcs::PhysicsSimulationComponent* sim = new Rcs::PhysicsSimulationComponent(simImpl);
    sim->setUpdateFrequency(1.0/bot.getConfig()->dt);
    bot.addHardwareComponent(sim);
    bot.setCallbackTriggerComponent(sim); // and it does drive the update loop
    
    // Add viewer component
#ifdef GRAPHICS_AVAILABLE
    Rcs::ViewerComponent* vc = NULL;
    Rcs::HUD* hud = NULL;
    if (!valgrind) {
        vc = new Rcs::ViewerComponent(bot.getGraph(), bot.getCurrentGraph(), true);
        hud = new Rcs::HUD(0, 0, 1024, 140);
        vc->getViewer()->add(hud);
        
        bot.getConfig()->initViewer(vc->getViewer());
        bot.addHardwareComponent(vc);
    }
#endif

    // Load control policy
    Rcs::ControlPolicy* controlPolicy = NULL;
    auto policyConfig = bot.getConfig()->properties->getChild("policy");
    if (policyConfig->exists()) {
        controlPolicy = Rcs::ControlPolicy::create(policyConfig);
        REXEC(1) {
            std::cout << "Loaded policy specified in the config file." << std::endl;
        }
    }
    else {
        REXEC(1) {
            std::cout << "Could not load a policy!" << std::endl;
        }
    }

//    controlPolicy = Rcs::loadMLPPolicyFromXml(policyFile.c_str(),
//            bot.getConfig()->observationModel->getDim(), bot.getConfig()
//            ->actionModel->getDim());
    
    // Start
    bot.startThreads();
    RMSG("Started robot.");
    bool startLoggerNextPolicyStart = false;
    
    // Main loop
    runLoop = true;
    RMSG("Main loop is running ...");
    while (runLoop) {
        // Check keys
#ifdef GRAPHICS_AVAILABLE
        if (vc && vc->getKeyCatcher()->getAndResetKey('q')) {
            runLoop = false;
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('l')) {
            if (bot.logger.isRunning()) {
                bot.logger.stop();
            }
            else if (bot.getControlPolicy() != controlPolicy) {
                // Defer until policy start
                startLoggerNextPolicyStart = true;
                RMSG("Deferring logger start until the policy is activated.");
            }
            else {
                bot.logger.start(bot.getConfig()->observationModel->getSpace(),
                                 bot.getConfig()->actionModel->getSpace(), 500);
            }
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('p')) {
            if (startLoggerNextPolicyStart) {
                bot.logger.start(bot.getConfig()->observationModel->getSpace(),
                                 bot.getConfig()->actionModel->getSpace(), 500);
            }
            bot.setControlPolicy(controlPolicy);
            RMSG("Control policy active.");
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('o')) {
            bot.setControlPolicy(NULL);
            RMSG("Holding initial state.");
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('b')) {
            // set ball position TODO
        }
        
        if (hud) {
            auto hudText = bot.getConfig()->getHUDText(
                simImpl->time(), bot.getObservation(), bot.getAction(), simImpl, ppmanager, nullptr);
            hud->setText(hudText);
        }
#endif

        // Wait a bit till next update
        Timer_waitDT(0.01);
    }
    
    // Terminate
    RMSG("Terminating...");
    bot.stopThreads();
    bot.disconnectCallback();
    
    delete controlPolicy;
    delete ppmanager;
    
    // Clean up global stuff. From the libxml2 documentation:
    // WARNING: if your application is multithreaded or has plugin support
    // calling this may crash the application if another thread or a plugin is
    // still using libxml2. It's sometimes very hard to guess if libxml2 is in
    // use in the application, some libraries or plugins may use it without
    // notice. In case of doubt abstain from calling this function or do it just
    // before calling exit() to avoid leak reports from valgrind !
    xmlCleanupParser();
    
    fprintf(stderr, "Thanks for using the Rcs libraries\n");
    return 0;
}



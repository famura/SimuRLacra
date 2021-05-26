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
#include <action/ActionModelIK.h>
#include <config/PropertySourceXml.h>
#include <control/ControlPolicy.h>
#include <control/ActionModelIKPolicy.h>
#include <observation/ObservationModel.h>
#include <physics/PhysicsParameterManager.h>

#include <Rcs_cmdLine.h>
#include <Rcs_macros.h>
#include <Rcs_parser.h>
#include <Rcs_resourcePath.h>
#include <Rcs_timer.h>
#include <Rcs_utils.h>
#include <KeyCatcherBase.h>
#include <PhysicsSimulationComponent.h>
#include <SegFaultHandler.h>
#include <TaskPosition3D.h>

#ifdef GRAPHICS_AVAILABLE

#include <PhysicsNode.h>
#include <ViewerComponent.h>

#endif

#include <random>

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
    Rcs::KeyCatcherBase::registerKey("h", "Deactivate policy and return to initial state");
    Rcs::KeyCatcherBase::registerKey("b", "Reset ball to a new random initial position");
    Rcs::KeyCatcherBase::registerKey("n", "Move the a to the position in front of the ball (pre-strike)");
    Rcs::KeyCatcherBase::registerKey("p", "Activate control policy (strike)");
    
    // Ctrl-C callback handler
    signal(SIGINT, quit);
    
    // This initialize the xml library and check potential mismatches between the version it was compiled for and
    // the actual shared library used.
    LIBXML_TEST_VERSION;
    
    // Parse command line arguments
    Rcs::CmdLineParser argP(argc, argv);
    char xmlFileName[128] = "ex_<ENVIRONMENT-NAME>_export.xml";
    char directory[128] = "../config/<ENVIRONMENT-FOLDER>";
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
    
    Rcs_addResourcePath("../config");
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
    //    sim->setSchedulingPolicy(SCHED_FIFO);
    
    bot.addHardwareComponent(sim);
    bot.setCallbackTriggerComponent(sim); // and it does drive the update loop

#ifdef GRAPHICS_AVAILABLE
    // Add viewer component
    Rcs::ViewerComponent* vc = nullptr;
    if (!valgrind) {
        //vc = new Rcs::ViewerComponent(bot.getGraph(), bot.getCurrentGraph(), true);
        vc = new Rcs::ViewerComponent(nullptr, nullptr, true);
        vc->getViewer()->add(new Rcs::PhysicsNode(simImpl));
        
        bot.getConfig()->initViewer(vc->getViewer());
        bot.addHardwareComponent(vc);
    }
#endif
    
    // Load control policy
    Rcs::ControlPolicy* controlPolicy = nullptr;
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
    
    // Start
    bot.startThreads();
    RMSG("Started robot.");
    bool startLoggerNextPolicyStart = false;
    
    // Main loop
    runLoop = true;
    RMSG("Main loop is running ...");
    while (runLoop) {

#ifdef GRAPHICS_AVAILABLE
        // Check if a key was pressed
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
                                 bot.getConfig()->actionModel->getSpace(), 5000);
            }
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('b')) {
            RcsBody* ball = RcsGraph_getBodyByName(simImpl->getGraph(), "Ball");
            if (ball) {
                // Set the ball to a random position
                std::random_device rd;  // used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); // standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<> distrX(0.48, 0.52);
                std::uniform_real_distribution<> distrY(1.3, 1.5);
                
                const double ballRBJAngles[6] = {distrX(gen), distrY(gen), ball->shape[0]->extents[0], 0, 0, 0};
                RcsGraph_setRigidBodyDoFs(simImpl->getGraph(), ball, ballRBJAngles);
                
                // Update the forward kinematics
                RcsGraph_setState(simImpl->getGraph(), simImpl->getGraph()->q,
                                  simImpl->getGraph()->q_dot);  // TODO @Michael: is this correct?
                
                REXEC(1) {
                    std::cout << "Set ball to new x, y position: " <<
                              ball->A_BI->org[0] << " " << ball->A_BI->org[1] << " [m]" << std::endl;
                }
            }
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('n')) {
            // Check if we are in the MiniGolfSim
            RcsBody* ball = RcsGraph_getBodyByName(simImpl->getGraph(), "Ball");
            RcsBody* clubTip = RcsGraph_getBodyByName(simImpl->getGraph(), "ClubTip");
            RcsBody* ground = RcsGraph_getBodyByName(simImpl->getGraph(), "Ground");
            
            if (dynamic_cast<Rcs::ActionModelIKPolicy*>(bot.getControlPolicy())) {
                RMSG("Already going to the ball...");
            }
            else if (ball != nullptr && clubTip != nullptr && ball != ground) {
                // Deactivate any policy
                bot.setControlPolicy(nullptr);
                
                Rcs::AMIKGeneric* amIK = new Rcs::AMIKGeneric(simImpl->getGraph());
                MatNd* fixedTaskValue = MatNd_create(3, 1);
                MatNd_set(fixedTaskValue, 0, 0, -0.0);
                MatNd_set(fixedTaskValue, 1, 0, -0.03);
                MatNd_set(fixedTaskValue, 2, 0, 0.00);
                amIK->addFixedTask(new Rcs::TaskPosition3D(simImpl->getGraph(), ball, clubTip, ground),
                                   fixedTaskValue);
                
                Rcs::ActionModelIKPolicy newPolicy = Rcs::ActionModelIKPolicy(amIK, bot.getConfig()->dt);
                
                RMSG("Going to a position in front of the ball...");
            }
            
            else {
                REXEC(2) {
                    std::cout << "Ignoring the 'n' key stroke" << std::endl;
                }
            }
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('p')) {
            if (startLoggerNextPolicyStart) {
                bot.logger.start(bot.getConfig()->observationModel->getSpace(),
                                 bot.getConfig()->actionModel->getSpace(), 5000);
            }
            controlPolicy->reset();
            bot.setControlPolicy(controlPolicy);
            RMSG("Control policy was reset and is active...");
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('o')) {
            bot.setControlPolicy(NULL);
            RMSG("Moving to initial state and holding it...");
        }
        
        auto hudText = bot.getConfig()->getHUDText(
            simImpl->time(), bot.getObservation(), bot.getAction(), simImpl, ppmanager, nullptr);
        vc->setText(hudText);
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

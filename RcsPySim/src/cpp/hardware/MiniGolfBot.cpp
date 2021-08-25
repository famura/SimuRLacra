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
#include <action/ActionModelIK.h>
#include <config/PropertySourceXml.h>
#include <control/ControlPolicy.h>
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
#include <LWAComponent.h>
#include <ViconComponent.h>

#ifdef GRAPHICS_AVAILABLE

#include <PhysicsNode.h>
#include <ViewerComponent.h>
#include <HUD.h>
#include <Rcs_Vec3d.h>

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
    std::cout << "Starting Rcs ..." << std::endl;
    
    Rcs::KeyCatcherBase::registerKey("q", "Quit");
    Rcs::KeyCatcherBase::registerKey("l", "Start/stop data logging");
    Rcs::KeyCatcherBase::registerKey("o", "Deactivate policy and go to home position (behind ball)");
    Rcs::KeyCatcherBase::registerKey("u", "Deactivate policy and go to a position up and above the ball");
    Rcs::KeyCatcherBase::registerKey("b", "Reset ball to a new random initial position");
    Rcs::KeyCatcherBase::registerKey("n", "Move the a to the pre-strike pose in front of the ball (q_des variant)");
    Rcs::KeyCatcherBase::registerKey("N", "Move the a to the pre-strike pose in front of the ball (policy variant)");
    Rcs::KeyCatcherBase::registerKey("p", "Activate control policy (strike)");
    
    // Ctrl-C callback handler
    signal(SIGINT, quit);
    
    // This initialize the xml library and check potential mismatches between the version it was compiled for and
    // the actual shared library used.
    LIBXML_TEST_VERSION;
    
    // Parse command line arguments
    Rcs::CmdLineParser argP(argc, argv);
    char xmlFileName[128] = "ex_config_export.xml";
    char directory[128] = "../config/<ENVIRONMENT-FOLDER>";
    argP.getArgument("-dl", &RcsLogLevel, "Debug level (default is 0)");
    argP.getArgument("-f", xmlFileName, "Configuration file name");
    argP.getArgument("-dir", directory, "Configuration file directory");
    bool runningOnRobot = argP.hasArgument("-real", "Run the program in the physical robot");
    bool valgrind = argP.hasArgument("-valgrind", "Start without GUIs and graphics");
    
    const char* hgr = getenv("SIT");
    if (hgr != nullptr) {
        std::string meshDir = std::string(hgr) + std::string("/Data/RobotMeshes/1.0/data");
        Rcs_addResourcePath(meshDir.c_str());
    }
    
    Rcs_addResourcePath("../config");
    Rcs_addResourcePath(directory);
    
    // Show help if requested
    if (argP.hasArgument("-h", "Show help message")) {
        Rcs::KeyCatcherBase::printRegisteredKeys();
        Rcs::CmdLineParser::print();
        Rcs_printResourcePath();
        return 0;
    }
    
    // Create simulated robot from config file
    std::cout << "Creating robot ..." << std::endl;
    Rcs::RcsPyBot bot(new Rcs::PropertySourceXml(xmlFileName));
    RLOG(0, "Current graph is 0x%x", bot.getCurrentGraph());
    
    bot.getConfig()->actionModel->reset();
    bot.getConfig()->observationModel->reset();
    
    Rcs::PhysicsParameterManager* ppmanager = nullptr;
    Rcs::PhysicsBase* simImpl = nullptr;
    Rcs::LWAComponent* lwa = nullptr;
    
    if (!runningOnRobot) {
        // Add physics simulator for testing
        ppmanager = bot.getConfig()->createPhysicsParameterManager();
        simImpl = ppmanager->createSimulator(bot.getConfig()->properties->getChild("initDomainParam"));
        
        Rcs::PhysicsSimulationComponent* sim = new Rcs::PhysicsSimulationComponent(simImpl);
        sim->setUpdateFrequency(1.0/bot.getConfig()->dt);
        //    sim->setSchedulingPolicy(SCHED_FIFO);
        bot.addHardwareComponent(sim);
        bot.setCallbackTriggerComponent(sim); // and it does drive the update loop
        
        // Add the VICON syste
        Rcs::ViconComponent* vic = new Rcs::ViconComponent();
        vic->setUpdateFrequency(100.);
        HTr viconOffs;
        HTr_setIdentity(&viconOffs);
        Vec3d_set(viconOffs.org, -0.4, -0.4, 0.85);
        vic->setOrigin(&viconOffs);
        bot.addHardwareComponent(vic);
        std::cout << "Added VICON component." << std::endl;
        
    }
    else {
        // Add the Schunk robot
        lwa = new Rcs::LWAComponent(bot.getCurrentGraph());
        lwa->setUpdateFrequency(100.);
        lwa->setSchedulingPolicy(SCHED_FIFO);
        lwa->setMotorLimits(Rcs::SchunkLWA::MotorSpeed_Fast);
        bot.addHardwareComponent(lwa);
        bot.setCallbackTriggerComponent(lwa);
        std::cout << "Added LWA component." << std::endl;
        
        // Add the VICON system
        Rcs::ViconComponent* vic = new Rcs::ViconComponent();
        vic->setUpdateFrequency(100.);
        vic->setSchedulingPolicy(SCHED_FIFO);
        HTr viconOffs;
        HTr_setIdentity(&viconOffs);
        Vec3d_set(viconOffs.org, -0.4, -0.4, 0.85);
        vic->setOrigin(&viconOffs);
        bot.addHardwareComponent(vic);
        std::cout << "Added VICON component." << std::endl;
//        bot.setCallbackTriggerComponent(vic); // and it does drive the update loop
    }

#ifdef GRAPHICS_AVAILABLE
    Rcs::Viewer* vc = nullptr;
    Rcs::KeyCatcher* kc = nullptr;
    Rcs::HUD* hud = nullptr;
    if (!valgrind) {
        vc = new Rcs::Viewer;
        kc = new Rcs::KeyCatcher;
        vc->add(kc);
        hud = new Rcs::HUD;
        vc->add(hud);
        
        if (!runningOnRobot) {
            auto pNode = new Rcs::PhysicsNode(simImpl);
            vc->add(pNode);
        }
        
        // Add the desired graph node of the action model
        auto nodeAMGraph = new Rcs::GraphNode(bot.getConfig()->actionModel->getGraph());
        nodeAMGraph->setGhostMode(true);
        vc->add(nodeAMGraph);
        
        // Optionally add the desired graph node of the IK-based action model
        Rcs::ActionModelIK* amIK = dynamic_cast<Rcs::ActionModelIK*>(bot.getConfig()->actionModel);
        if (amIK) {
            auto nodeAMDesGraph = new Rcs::GraphNode(amIK->getDesiredGraph());
            nodeAMDesGraph->setGhostMode(true, "RED");
            vc->add(nodeAMDesGraph);
        }
        
        auto gNodeCurr = new Rcs::GraphNode(bot.getCurrentGraph());
        gNodeCurr->setGhostMode(true, "BLUE");
        vc->add(gNodeCurr);
        
        auto gNodeDes = new Rcs::GraphNode(bot.getGraph());
        gNodeDes->setGhostMode(true, "GREEN");
        vc->add(gNodeDes);
        vc->runInThread();
    }
#endif
    
    // Sync graphs
    bot.syncGraphsToCurrent();
    bot.updateGraph();
    
    bot.getConfig()->actionModel->reset();
    bot.getConfig()->observationModel->reset();
    
    // Load the (learned) control policy
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
    
    // Load additional (optional) policies
    Rcs::ControlPolicy* preStrikePolicy = nullptr;
    auto preStrikePolicyConfig = bot.getConfig()->properties->getChild("preStrikePolicy");
    if (preStrikePolicyConfig->exists()) {
        preStrikePolicy = Rcs::ControlPolicy::create(preStrikePolicyConfig);
        REXEC(1) {
            std::cout << "Loaded pre-strike policy specified in the config file." << std::endl;
        }
    }
    
    // Start
    bot.startThreads();
    std::cout << "Started robot." << std::endl;
    bool startLoggerNextPolicyStart = false;
    
    sleep(0.5);
    
    // Sync graphs
    bot.syncGraphsToCurrent();
    bot.updateGraph();
    
    bot.getConfig()->actionModel->reset();//TODO check
    bot.getConfig()->observationModel->reset();
    
    if (runningOnRobot)
        RPAUSE_MSG("Now be careful - robot moves");
    
    
    if (lwa) {
        lwa->setWriteCommands(true);
    }
    
    // Main loop
    runLoop = true;
    std::cout << "Main loop is running ..." << std::endl;
    while (runLoop) {

#ifdef GRAPHICS_AVAILABLE
        // Check if a key was pressed
        if (vc && kc->getAndResetKey('q')) {
            runLoop = false;
        }
        
        if (vc && kc->getAndResetKey('S')) {
            // Recreate the action model to sync the graphs
            bot.getConfig()->actionModel = bot.getConfig()->createActionModel();
            std::cout << "Graphs synced!" << std::endl;
        }
        
        if (vc && kc->getAndResetKey('l')) {
            if (bot.logger.isRunning()) {
                bot.logger.stop();
            }
            else if (bot.getControlPolicy() != controlPolicy) {
                // Defer until policy start
                startLoggerNextPolicyStart = true;
                REXEC(1) { std::cout << "Deferring logger start until the policy is activated." << std::endl; }
            }
            else {
                bot.logger.start(bot.getConfig()->observationModel->getSpace(),
                                 bot.getConfig()->actionModel->getSpace(), 800);
            }
        }
        
        if (!runningOnRobot && vc && kc->getAndResetKey('b')) {
            RcsBody* ball = RcsGraph_getBodyByName(bot.getCurrentGraph(), "Ball");
            if (ball) {
                // Set the ball to the fixed initial position
                const double ballRBJAngles[6] = {0.3, 1.3, ball->shape[0]->extents[0], 0, 0, 0};
                RcsGraph_setRigidBodyDoFs(bot.getCurrentGraph(), ball, ballRBJAngles);
                
                // Reset the physics simulation
                simImpl->reset(bot.getCurrentGraph()->q);
                
                REXEC(1) {
                    std::cout << "Set ball to new x, y position: " <<
                              ball->A_BI->org[0] << " " << ball->A_BI->org[1] << " [m]" << std::endl;
                }
            }
        }
        if (!runningOnRobot && vc && kc->getAndResetKey('B')) {
            RcsBody* ball = RcsGraph_getBodyByName(bot.getCurrentGraph(), "Ball");
            if (ball) {
                // Set the ball to a random position
                std::random_device rd;  // used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); // standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<> distrX(0.28, 0.32);
                std::uniform_real_distribution<> distrY(1.29, 1.31);
                
                const double ballRBJAngles[6] = {distrX(gen), distrY(gen), ball->shape[0]->extents[0], 0, 0, 0};
                RcsGraph_setRigidBodyDoFs(bot.getCurrentGraph(), ball, ballRBJAngles);
                
                // Reset the physics simulation
                simImpl->reset(bot.getCurrentGraph()->q);
                
                REXEC(1) {
                    std::cout << "Set ball to new x, y position: " <<
                              ball->A_BI->org[0] << " " << ball->A_BI->org[1] << " [m]" << std::endl;
                }
            }
        }
        if (vc && kc->getAndResetKey('N')) {
            // Freeze
            bot.getConfig()->actionModel->setGraph(RcsGraph_clone(bot.getCurrentGraph()));
            
            // Check if we are in the MiniGolfSim
            RcsBody* ball = RcsGraph_getBodyByName(bot.getCurrentGraph(), "Ball");
            RcsBody* clubTip = RcsGraph_getBodyByName(bot.getCurrentGraph(), "ClubTip");
            RcsBody* ground = RcsGraph_getBodyByName(bot.getCurrentGraph(), "Ground");
            auto amIK = dynamic_cast<Rcs::AMIKGeneric*>(bot.getConfig()->actionModel);
            
            if (ball != nullptr && clubTip != nullptr && ground != nullptr && amIK != nullptr) {
                // Set the control policy active
                preStrikePolicy->reset();
                bot.setControlPolicy(preStrikePolicy);
                REXEC(0) { std::cout << "Going to a position in front of the ball ..." << std::endl; }
            }
            
            else {
                REXEC(2) { std::cout << "Ignoring the 'n' key stroke" << std::endl; }
            }
        }
        
        if (vc && kc->getAndResetKey('n')) {
            // Command the pre-strike pose (specified by overwriting some joint values of the desired graph)
            MatNd* q_des_prestrike = MatNd_clone(bot.getGraph()->q);
            q_des_prestrike->ele[1] = RCS_DEG2RAD(18.996253);
            q_des_prestrike->ele[2] = RCS_DEG2RAD(-87.227101);
            q_des_prestrike->ele[3] = RCS_DEG2RAD(74.149568);
            q_des_prestrike->ele[4] = RCS_DEG2RAD(-75.577025);
            q_des_prestrike->ele[5] = RCS_DEG2RAD(56.207369);
            q_des_prestrike->ele[6] = RCS_DEG2RAD(-175.162794);
            q_des_prestrike->ele[7] = RCS_DEG2RAD(-41.543793);
            
            bot.setControlPolicy(nullptr, q_des_prestrike);
            MatNd_destroy(q_des_prestrike);
            REXEC(0) { std::cout << "Moving to pre-initial state and holding it ..." << std::endl; }
        }
        
        if (vc && kc->getAndResetKey('k')) {
            bot.logger.start(bot.getConfig()->observationModel->getSpace(),
                             bot.getConfig()->actionModel->getSpace(), 800);
            
            // Command the strike (specified by overwriting some joint values of the desired graph)
            MatNd* q_des_strike = MatNd_clone(bot.getGraph()->q);
            q_des_strike->ele[1] = RCS_DEG2RAD(8.628977);
            q_des_strike->ele[2] = RCS_DEG2RAD(-93.443498);
            q_des_strike->ele[3] = RCS_DEG2RAD(72.302435);
            q_des_strike->ele[4] = RCS_DEG2RAD(-82.31844);
            q_des_strike->ele[5] = RCS_DEG2RAD(52.146531);
            q_des_strike->ele[6] = RCS_DEG2RAD(-183.896354);
            q_des_strike->ele[7] = RCS_DEG2RAD(-51.560886);
            
            bot.setControlPolicy(nullptr, q_des_strike);
            MatNd_destroy(q_des_strike);
            REXEC(0) { std::cout << "Strike and hold ..." << std::endl; }
        }
        
        if (vc && kc->getAndResetKey('p')) {
            // Freeze
            bot.getConfig()->actionModel->setGraph(RcsGraph_clone(bot.getCurrentGraph()));
            
            // Start the logger now if desired
            if (startLoggerNextPolicyStart) {
                bot.logger.start(bot.getConfig()->observationModel->getSpace(),
                                 bot.getConfig()->actionModel->getSpace(), 800);
            }
            
            // Set the control policy active
            controlPolicy->reset();
            bot.setControlPolicy(controlPolicy);
            REXEC(0) { std::cout << "Control policy was reset and is active ..." << std::endl; }
        }
        
        if (vc && kc->getAndResetKey('u')) {
            // Command an initial pose above and in front of the ball (specified by overwriting some joint values of
            // the desired graph)
            MatNd* q_des_upfront = MatNd_clone(bot.getGraph()->q);
            q_des_upfront->ele[1] = RCS_DEG2RAD(-17.024125);
            q_des_upfront->ele[2] = RCS_DEG2RAD(-107.651808);
            q_des_upfront->ele[3] = RCS_DEG2RAD(65.137897);
            q_des_upfront->ele[4] = RCS_DEG2RAD(-93.876418);
            q_des_upfront->ele[5] = RCS_DEG2RAD(41.902234);
            q_des_upfront->ele[6] = RCS_DEG2RAD(-186.725503);
            q_des_upfront->ele[7] = RCS_DEG2RAD(-29.29585);
            bot.setControlPolicy(nullptr, q_des_upfront);
            MatNd_destroy(q_des_upfront);
            REXEC(1) { std::cout << "Moving to pre-initial state and holding it ..." << std::endl; }
        }
        
        if (vc && kc->getAndResetKey('o')) {
            // Command the initial pose (specified by overwriting some joint values of the desired graph)
            MatNd* q_des_home = MatNd_clone(bot.getGraph()->q);
            q_des_home->ele[1] = RCS_DEG2RAD(29.821247);
            q_des_home->ele[2] = RCS_DEG2RAD(-86.89066);
            q_des_home->ele[3] = RCS_DEG2RAD(88.293785);
            q_des_home->ele[4] = RCS_DEG2RAD(-66.323556);
            q_des_home->ele[5] = RCS_DEG2RAD(63.39102);
            q_des_home->ele[6] = RCS_DEG2RAD(-148.848292);
            q_des_home->ele[7] = RCS_DEG2RAD(-11.296764);
            bot.setControlPolicy(nullptr, q_des_home);
            MatNd_destroy(q_des_home);
            REXEC(1) { std::cout << "Moving to initial state and holding it ..." << std::endl; }
        }
        
        std::string hudText = "a";
        if (!runningOnRobot) {
            hudText = bot.getConfig()->getHUDText(
                simImpl->time(), bot.getObservation(), bot.getAction(), simImpl, ppmanager, nullptr);
        }
        else {
            hudText = bot.getConfig()->getHUDText(
                // TODO how to get the time form the real robot?
                0, bot.getObservation(), bot.getAction(), nullptr, ppmanager, nullptr);
        }
        hud->setText(hudText);
#endif
        
        const RcsBody* ball = RcsGraph_getBodyByName(bot.getCurrentGraph(), "Ball");
        RCHECK(ball);
        RLOG(5, "ball body (curr graph): %f %f %f", ball->A_BI->org[0], ball->A_BI->org[1], ball->A_BI->org[2]);
        
        // Wait a bit till next update
        Timer_waitDT(0.01);
    }
    
    // Terminate
    std::cout << "Terminating ..." << std::endl;
    bot.stopThreads();
    bot.disconnectCallback();
    
    delete controlPolicy;
    delete ppmanager;
    
    // Clean up global stuff. From the libxml2 documentation:
    // WARNING: if your application is multi-threaded or has plugin support
    // calling this may crash the application if another thread or a plugin is
    // still using libxml2. It's sometimes very hard to guess if libxml2 is in
    // use in the application, some libraries or plugins may use it without
    // notice. In case of doubt abstain from calling this function or do it just
    // before calling exit() to avoid leak reports from valgrind !
    xmlCleanupParser();
    
    std::cerr << "Thanks for using the Rcs libraries" << std::endl;
    return 0;
}

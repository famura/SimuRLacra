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

#include "PolicyComponent.h"

#include "EntityBase.h"
#include "GraphComponent.h"
#include "GraphicsWindow.h"
#include "PhysicsComponent.h"
#include "IKComponent.h"
#include "EventGui.h"
#include "EcsHardwareComponents.h"
//#include "WheelPlannerComponent.h"
//#include "WheelStrategy7D.h"
//#include "WheelConstraint.h"
//#include "WheelNode.h"
//#include "WheelObjectModel.h"
#include "TTSComponent.h"

#include <Rcs_cmdLine.h>
#include <Rcs_resourcePath.h>
#include <Rcs_guiFactory.h>
#include <Rcs_graph.h>
#include <KeyCatcherBase.h>
#include <SegFaultHandler.h>
#include <config/PropertySourceXml.h>
#include <Rcs_timer.h>


RCS_INSTALL_ERRORHANDLERS


using namespace Rcs;


static bool runLoop = true;

/*******************************************************************************
 *
 ******************************************************************************/
void quit()
{
    RLOG(0, "Quit::quit()");
    runLoop = false;
}


static void runPolicy()
{
    // load config
    CmdLineParser argP;
    double ttc = 8.0, alpha = 0.02;
    unsigned int speedUp = 1, loopCount = 0;
    size_t queueSize = 0;
    char cfgFile[64] = "ex_UNSPECIFIED.xml";
    char physicsEngine[32] = "Vortex";
    char physicsCfg[128] = "config/physics/physics.xml";
    argP.getArgument("-f", cfgFile, "Experiment configuration file name");
    argP.getArgument("-speedUp", &speedUp, "Speed-up factor (default is %d)",
                     speedUp);
    argP.getArgument("-physicsEngine", physicsEngine,
                     "Physics engine (default is \"%s\")", physicsEngine);
    argP.getArgument("-physics_config", physicsCfg, "Configuration file name"
                                                    " for physics (default is %s)", physicsCfg);
    argP.getArgument("-ttc", &ttc, "Transition time (default is %f)", ttc);
    argP.getArgument("-alpha", &alpha, "Null space scaler (default is %f)", alpha);
    bool noEventGui = argP.hasArgument("-noEventGui", "Don't launch EventGui");
    bool seqSim = argP.hasArgument("-sequentialPhysics", "Physics simulation step"
                                                         "in updateGraph()");
    bool seqViewer = argP.hasArgument("-sequentialViewer", "Viewer frame call "
                                                           "in \"Render\" event");
    bool pause = argP.hasArgument("-pause", "Pause after each process() call");
    bool sync = argP.hasArgument("-sync", "Run as sequential as possible");
    bool simpleGraphics = argP.hasArgument("-simpleGraphics", "OpenGL without "
                                                              "shadows and anti-aliasing");
    bool noSpeedCheck = argP.hasArgument("-nospeed", "Disable speed limit checks");
    bool noJointCheck = argP.hasArgument("-nojl", "Disable joint limit checks");
    bool noCollCheck = argP.hasArgument("-nocoll", "Disable collision checks");
    
    if (sync) {
        seqSim = true;
        seqViewer = true;
    }
    
    EntityBase entity;
    if (pause) {
        entity.call("TogglePause");
    }
    
    // Register lifecycle events
    auto updateGraph = entity.registerEvent<RcsGraph*>("UpdateGraph");
    auto computeKinematics = entity.registerEvent<RcsGraph*>("ComputeKinematics");
    auto postUpdateGraph = entity.registerEvent<RcsGraph*, RcsGraph*>("PostUpdateGraph");
    auto updatePolicy = entity.registerEvent<const RcsGraph*>("UpdatePolicy");
    auto setJointCommand = entity.registerEvent<const MatNd*>("SetJointCommand");
    auto setRenderCommand = entity.registerEvent<>("Render");
    auto setComplianceCommand = entity.registerEvent<std::vector<double>>("SetComplianceCommand");
    
    entity.registerEvent<>("Start");
    entity.registerEvent<>("Stop");
    entity.registerEvent<>("EmergencyStop");
    entity.registerEvent<>("EmergencyRecover");
    entity.registerEvent<>("Quit");
    entity.subscribe("Quit", &quit);
    
    // setup policy component, which will load the experiment and policy data
    PolicyComponent policy(&entity, new PropertySourceXml(cfgFile), false);
//    policy.setSpeedLimitCheck(!noSpeedCheck);
    policy.setJointLimitCheck(!noJointCheck);
    policy.setCollisionCheck(!noCollCheck);
    RcsGraph* graph = policy.getExperiment()->graph;
    
    // setup viewer and graph components
    GraphicsWindow viewer(&entity, false, seqViewer, simpleGraphics);
    GraphComponent graphC(&entity, graph);
    
    // setup hardware
    std::vector<ComponentBase*> hwc = getHardwareComponents(entity, graph);
    
    if (hwc.empty()) {
        // no hardware set, use physics simulator
//        RCSGRAPH_TRAVERSE_JOINTS(graph) {
//            JNT->ctrlType = RCSJOINT_CTRL_POSITION;
//        }
        
        auto pc = new PhysicsComponent(&entity, graph, physicsEngine,
                                       physicsCfg, !seqSim);
        
        if (seqSim && seqViewer) {
#if defined (USE_BULLET)
            osg::ref_ptr<BulletDebugNode> btNd =
        new BulletDebugNode(pc->getPhysicsSimulation(), viewer.getFrameMtx());
      viewer.add(btNd.get());
#endif
        }
        
        hwc.push_back(pc);
    }
    
    // Setup viewer key callbacks
    viewer.setKeyCallback('d', [&entity](char k) {
        RLOG(0, "Reseting IME failure flag");
        entity.publish("ResetErrorFlag");
    }, "Reseting IME failure flag");
    
    viewer.setKeyCallback('m', [&entity, ttc](char k) {
        RLOG(0, "Move robot to initial pose");
        entity.publish("InitializeMovement", ttc);
    }, "Move robot to initial pose");
    
    // End of configuration, print help if requested
    if (argP.hasArgument("-h")) {
        entity.print(std::cout);
        return;
    }
    
    // Setup event GUI if requested, only do so now since all events are registered
    if (!noEventGui) {
        EventGui::create(&entity);
    }
    
    
    // Start threads (if any)
    RPAUSE_MSG_DL(1, "Start");
    entity.publish("Start");
    int nIter = entity.processUntilEmpty(10);
    RLOG(1, "Start took %d process() calls, queue is %zu",
         nIter, entity.queueSize());
    
    // Initialization sequence
    entity.publish("Render");
    nIter = entity.processUntilEmpty(10);
    RLOG(1, "Render took %d process() calls, queue is %zu",
         nIter, entity.queueSize());
    
    RPAUSE_MSG_DL(1, "updateGraph");
    updateGraph->call(graphC.getGraph());
    nIter = entity.processUntilEmpty(10);
    RLOG(1, "updateGraph took %d process() calls, queue is %zu",
         nIter, entity.queueSize());
    
    updatePolicy->call(graphC.getGraph());
    entity.processUntilEmpty(10);
    entity.publish("Render");
    entity.processUntilEmpty(10);
    
    RPAUSE_MSG_DL(1, "updateGraph");
    updateGraph->call(graphC.getGraph());
    nIter = entity.processUntilEmpty(10);
    RLOG(1, "updateGraph took %d process() calls, queue is %zu",
         nIter, entity.queueSize());
    
    updatePolicy->call(graphC.getGraph());
    entity.processUntilEmpty(10);
    
    entity.publish<const RcsGraph*>("InitFromState", graphC.getGraph());
    entity.publish("Render");
    entity.processUntilEmpty(10);

//    Rcs::GraphNode* gn = viewer.getGraphNodeById("Physics");
//    if (gn)
//    {
//        Rcs::BodyNode* anchor = gn->getBodyNode("WheelPole");
//        if (anchor)
//        {
//            anchor->addChild(new Rcs::WheelNode(dmpc.getStrategy()));
//            entity.publish("Render");
//            entity.processUntilEmpty(10);
//        }
//    }
    
    
    RPAUSE_MSG_DL(1, "EnableCommands");
    entity.publish("EnableCommands");
    nIter = entity.processUntilEmpty(10);
    RLOG(1, "InitFromState++ took %d process() calls, queue is %zu",
         nIter, entity.queueSize());
    RPAUSE_MSG_DL(1, "Enter runLoop");
    
    
    while (runLoop) {
        double dtProcess = Timer_getSystemTime();
        updateGraph->call(graphC.getGraph());
        
        // Update graph state
        computeKinematics->call(graphC.getGraph());
        
        postUpdateGraph->call(policy.getDesiredGraph(), graphC.getGraph());
        
        // compute input from policy
        updatePolicy->call(graphC.getGraph());
        
        // This is called right before sending the commands
        // checkEmergencyStop->call();
        
        // Distribute the IKComponents internal command to all motor components
        setJointCommand->call(policy.getJointCommandPtr());   // motors <- q

//        setComplianceCommand->call(dmpc.getComplianceWrench());
        
        // This triggers graphics updates in some components
        setRenderCommand->call();
        
        queueSize = std::max(entity.queueSize(), queueSize);
        
        entity.process();
        entity.stepTime();
        
        dtProcess = Timer_getSystemTime() - dtProcess;
        
        
        char text[256];
        snprintf(text, 256, "Time: %.3f   dt: %.1f msec (< %.1f msec)   queue: %zu",
                 entity.getTime(), 1.0e3*dtProcess, 1.0e3*entity.getDt(), queueSize);
        entity.publish("SetTextLine", std::string(text), 1);
        
        entity.publish("SetTextLine", policy.getStateText(), 2);
        
        std::vector<std::string> hudLines;
        policy.getExperiment()->getHUDText(hudLines, entity.getTime(), policy.getObservation(), policy.getAction(),
                                           NULL, NULL, NULL);
        for (size_t i = 0; i < hudLines.size(); ++i) {
            entity.publish("SetTextLine", hudLines[i], (int) i + 3);
        }

//        snprintf(text, 256, "Motion time: %5.3f\nState: ",
//                 dmpc.getMotionEndTime());
//        std::vector<int> searchState = dmpc.getState();
//        for (size_t i = 0; i < searchState.size(); ++i)
//        {
//            char a[8];
//            snprintf(a, 8, "%d ", (int)searchState[i]);
//            strcat(text, a);
//        }
//
//        entity.publish("SetTextLine", std::string(text), 2);
        
        loopCount++;
        if (loopCount%speedUp == 0) {
            Timer_waitDT(entity.getDt() - dtProcess);
        }
        
    }
    
    entity.publish<>("Stop");
    entity.process();
    
    for (size_t i = 0; i < hwc.size(); ++i) {
        delete hwc[i];
    }
}

/*******************************************************************************
 *
 ******************************************************************************/
int main(int argc, char** argv)
{
    char directory[128] = "config/UNSPECIFIED";
    int mode = 1;
    CmdLineParser argP(argc, argv);
    argP.getArgument("-dl", &RcsLogLevel, "Debug level (default is %d)",
                     RcsLogLevel);
    argP.getArgument("-dir", directory, "Configuration file directory "
                                        "(default is %s)", directory);
    argP.getArgument("-m", &mode, "Test mode (default is %d)", mode);
    
    Rcs_addResourcePath(RCS_CONFIG_DIR);
    Rcs_addResourcePath("config");
    Rcs_addResourcePath(directory);
    
    switch (mode) {
        case 1:
            runPolicy();
            break;
        
        default:
            RMSG("No mode %d", mode);
    }
    
    
    if (argP.hasArgument("-h")) {
        Rcs_printResourcePath();
        Rcs::KeyCatcherBase::printRegisteredKeys();
        argP.print();
//        printf("\nTest modes:\n\t1:\tSimple planning\n\t2:\tTrajectory test\n\n");
    }
    
    
    RcsGuiFactory_shutdown();
    xmlCleanupParser();
    
    fprintf(stderr, "Thanks for using the Rcs libraries\n");
    
    return 0;
}

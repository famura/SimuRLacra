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

#include "ExperimentConfig.h"
#include "action/AMNormalized.h"
#include "observation/ObservationModel.h"
#include "observation/OMNormalized.h"
#include "observation/OMPartial.h"
#include "physics/PhysicsParameterManager.h"

#include <Rcs_resourcePath.h>
#include <Rcs_macros.h>
#include <Rcs_basicMath.h>

#include <map>
#include <string>
#include <stdexcept>
#include <sstream>
#include <Rcs_parser.h>
#include <Rcs_collisionModel.h>

namespace Rcs
{

// Utilities for collision model
static inline void
copy_prop_to_xml_attr(xmlNodePtr node, PropertySource* source, const char* name, const char* xmlName = nullptr)
{
    if (xmlName == nullptr) {
        xmlName = name;
    }
    // Get string value from source. Falls back to xmlName to allow direct copies. Some of the names are just atrocious.
    std::string value;
    if (source->getProperty(value, name) || source->getProperty(value, xmlName)) {
        // Set to node
        xmlSetProp(node, BAD_CAST xmlName, BAD_CAST value.c_str());
    }
}

RcsCollisionMdl* RcsCollisionModel_createFromConfig(RcsGraph* graph, PropertySource* config)
{
    // Check for filename
    std::string file;
    if (config->getProperty(file, "file")) {
        
        char txt[256];
        bool fileExists = Rcs_getAbsoluteFileName(file.c_str(), txt);
        
        if (!fileExists) {
            RMSG("Resource path is:");
            Rcs_printResourcePath();
            
            std::ostringstream os;
            os << "Could not find collision model file: " << file;
            throw std::invalid_argument(os.str());
        }
        
        // Load from xml file
        xmlDocPtr doc = nullptr;
        xmlNodePtr node = parseXMLFile(txt, "CollisionModel", &doc);
        if (!node) {
            throw std::invalid_argument("Error parsing collision model XML.");
        }
        
        // Create the object
        auto collisionMdl = RcsCollisionModel_createFromXML(graph, node);
        // free xml
        xmlFreeDoc(doc);
        
        if (!collisionMdl) {
            throw std::invalid_argument("Error in collision model configuration! (see stderr)");
        }
        return collisionMdl;
    }
    
    // Convert collision config to XML.
    // While this is kind of unnecessary if config is backed by XML, we cannot do this generically for python.
    xmlDocPtr doc = xmlNewDoc(nullptr);
    xmlNodePtr cmNode = xmlNewDocNode(doc, nullptr, BAD_CAST "CollisionModel", nullptr);
    xmlDocSetRootElement(doc, cmNode);
    
    copy_prop_to_xml_attr(cmNode, config, "threshold");
    copy_prop_to_xml_attr(cmNode, config, "mixtureCost", "MixtureCost");
    copy_prop_to_xml_attr(cmNode, config, "penetrationSlope", "PenetrationSlope");
    
    // Convert reference to pointer to allow both 'pairs' and 'CollisionPair' child list.
    // The latter is for xml compatibility.
    auto pairs = &config->getChildList("pairs");
    if (pairs->empty()) {
        pairs = &config->getChildList("CollisionPair");
    }
    for (auto child : *pairs) {
        xmlNodePtr pairNode = xmlNewDocNode(doc, nullptr, BAD_CAST "CollisionPair", nullptr);
        xmlAddChild(cmNode, pairNode);
        
        copy_prop_to_xml_attr(pairNode, child, "body1");
        copy_prop_to_xml_attr(pairNode, child, "body2");
        
        copy_prop_to_xml_attr(pairNode, child, "threshold");
        copy_prop_to_xml_attr(pairNode, child, "weight");
    }
    
    // Create the object
    auto collisionMdl = RcsCollisionModel_createFromXML(graph, cmNode);
    
    // Free temporary xml
    xmlFreeDoc(doc);
    
    // Throw exception if collision model failed to create
    if (!collisionMdl) {
        throw std::invalid_argument("Error in collision model configuration! (see stderr)");
    }
    return collisionMdl;
}

// Experiment type registry
static std::map<std::string, ExperimentConfig::ExperimentConfigCreateFunction> registry;

void ExperimentConfig::registerType(const char* name, ExperimentConfig::ExperimentConfigCreateFunction creator)
{
    // Store in registry
    registry[name] = creator;
}

ExperimentConfig* ExperimentConfig::create(PropertySource* properties)
{
    ExperimentConfig* result;
    try {
        // Get experiment name
        std::string key;
        if (!properties->getProperty(key, "envType")) {
            throw std::invalid_argument("Property source is missing 'envType' entry.");
        }
        
        // Look up factory
        auto iter = registry.find(key);
        if (iter == registry.end()) {
            std::ostringstream os;
            os << "Unknown experiment type '" << key << "'.";
            throw std::invalid_argument(os.str());
        }
        
        // Create instance
        result = iter->second();
        
    } catch (...) {
        // Up to here, this method owns properties, so we must delete them on error
        delete properties;
        // The last part that could throw is the creation of result, so we don't need to delete it
        throw;
    }
    
    // Load transfers ownership of properties to result
    try {
        result->load(properties);
        return result;
    } catch (...) {
        // Error, make sure to delete result (will also delete properties)
        delete result;
        throw;
    }
}

ExperimentConfig::ExperimentConfig()
{
    properties = nullptr;
    actionModel = nullptr;
    observationModel = nullptr;
    dt = 0.01;
}

ExperimentConfig::~ExperimentConfig()
{
    // Delete stored models
    delete observationModel;
    delete actionModel;
    
    RcsCollisionModel_destroy(collisionMdl);
    // Destroy graph
    RcsGraph_destroy(graph);
    
    // Delete property source
    delete properties;
}

void ExperimentConfig::load(PropertySource* properties)
{
    this->properties = properties;
    
    // Init seed
    int seed;
    if (properties->getProperty(seed, "seed")) {
        Math_srand48(seed);
    }
    
    // Init Rcs config dir
    Rcs_addResourcePath(RCS_CONFIG_DIR);
    // Read config dir from properties
    std::string extraConfigDir;
    if (properties->getProperty(extraConfigDir, "extraConfigDir")) {
        Rcs_addResourcePath(extraConfigDir.c_str());
    }
    
    // Init graph (this is NOT the graph form the physics simulation)
    std::string graphFileName = "gScenario.xml";
    properties->getProperty(graphFileName, "graphFileName");
    graph = RcsGraph_create(graphFileName.c_str());
    if (!graph) {
        throw std::invalid_argument("Graph not found: " + graphFileName);
    }
    
    // Load collision model - action and observation models might use it
    auto collCfg = properties->getChild("collisionConfig");
    if (collCfg->exists()) {
        collisionMdl = RcsCollisionModel_createFromConfig(graph, collCfg);
    }
    else {
        collisionMdl = nullptr;
    }
    
    // Create action model (the graph is specified in the associated experiment config file)
    actionModel = createActionModel();
    RCHECK(actionModel);
    
    // Add normalized action model if desired
    if (properties->getPropertyBool("normalizeActions", false)) {
        // The temp var suppresses a UNINIT.HEAP.MUST from klocwork
        auto inner = actionModel;
        actionModel = new AMNormalized(inner);
    }
    
    observationModel = createObservationModel();
    RCHECK(observationModel);
    // Add normalized observation model if desired
    if (properties->getPropertyBool("normalizeObservations")) {
        // Load bound overrides
        PropertySource* minOverride = properties->getChild("obsNormOverrideLower");
        PropertySource* maxOverride = properties->getChild("obsNormOverrideUpper");
        
        // The temp var suppresses a wrong UNINIT.HEAP.MUST from klocwork
        auto inner = observationModel;
        observationModel = new OMNormalized(inner, minOverride, maxOverride);
    }

    // Make observation partial if desired
    auto partialObs = properties->getChild("partialObservation");
    if (partialObs) {
        bool exclude = partialObs->getPropertyBool("exclude");
        std::vector<std::string> names;
        if (partialObs->getProperty(names, "names")) {
            auto inner = observationModel;
            bool autoSelectVelocity = partialObs->getPropertyBool("autoSelectVelocity");
            observationModel = OMPartial::fromNames(inner, names, exclude, autoSelectVelocity);
        }
    }
    
    // Load additional common properties
    properties->getProperty(dt, "dt");
    checkJointLimits = properties->getPropertyBool("checkJointLimits", true);
}

ObservationModel* ExperimentConfig::createObservationModel()
{
    // Return nullptr by default to use state model
    return nullptr;
}

InitStateSetter* ExperimentConfig::createInitStateSetter()
{
    // Return nullptr by default to use graph state from xml
    return nullptr;
}

ForceDisturber* ExperimentConfig::createForceDisturber()
{
    // Return nullptr by default to skip
    return nullptr;
}

PhysicsParameterManager* ExperimentConfig::createPhysicsParameterManager()
{
    // read general physics properties
    std::string physicsEngine = "Bullet";
    std::string physicsConfigFile = "physics/physics.xml";
    
    properties->getProperty(physicsEngine, "physicsEngine");
    properties->getProperty(physicsConfigFile, "physicsConfigFile");
    
    // Create manager and populate it
    PhysicsParameterManager* manager = new PhysicsParameterManager(graph, physicsEngine, physicsConfigFile);
    populatePhysicsParameters(manager);
    
    // Try to create simulator to catch config errors eagerly
    PhysicsBase* testSim = manager->createSimulator(PropertySource::empty());
    if (testSim) {
        // Works, delete it again
        delete testSim;
    }
    else {
        // Does not work, abort here and report error
        delete manager;
        std::ostringstream os;
        os << "Unable to create physics simulator using " << physicsEngine << " engine.";
        throw std::invalid_argument(os.str());
    }
    
    return manager;
}

void ExperimentConfig::initViewer(Rcs::Viewer* viewer)
{
    // Do nothing by default
}

void ExperimentConfig::populatePhysicsParameters(PhysicsParameterManager* manager)
{
    // Do nothing by default
}

void ExperimentConfig::getHUDText(
    std::vector<std::string>& linesOut, double currentTime,
    const MatNd* currentObservation, const MatNd* currentAction,
    PhysicsBase* simulator, PhysicsParameterManager* physicsManager,
    ForceDisturber* forceDisturber)
{
    // Obtain simulator name
    const char* simname = "None";
    if (simulator != nullptr) {
        simname = simulator->getClassName();
    }
    
    char hudText[2048];
    sprintf(hudText, "physics engine: %s        simulation time:             %2.3f s", simname, currentTime);
    linesOut.emplace_back(hudText);
}

std::string ExperimentConfig::getHUDText(
    double currentTime, const MatNd* currentObservation,
    const MatNd* currentAction, PhysicsBase* simulator,
    PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber)
{
    // Get lines using getHUDText function for the specific ExperimentConfig
    std::vector<std::string> lines;
    getHUDText(lines, currentTime, currentObservation, currentAction, simulator, physicsManager, forceDisturber);
    
    // Concat the lines
    std::ostringstream os;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) { os << '\n'; }
        os << lines[i];
    }
    
    return os.str();
}
    
} /* namespace Rcs */

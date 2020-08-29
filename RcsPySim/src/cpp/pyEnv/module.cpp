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

#include "RcsSimEnv.h"
#include "config/PropertySourceDict.h"
#include "config/PropertySourceXml.h"
#include "control/MLPPolicy.h"
#include "physics/vortex_log.h"
#include "util/BoxSpace.h"
#include "util/type_casters.h"
#include "util/pybind_dict_utils.h"

#include <pybind11/stl.h>

namespace py = pybind11;

#include <Rcs_resourcePath.h>
#include <Rcs_macros.h>
#include <Rcs_MatNd.h>
#include <Rcs_Vec3d.h>
#include <PhysicsFactory.h>

#include <SegFaultHandler.h>
#include <Rcs_typedef.h>

RCS_INSTALL_ERRORHANDLERS

//void define_gui_classes(py::module& m);

PYBIND11_MODULE(_rcsenv, m)
{
    // define exceptions
    py::register_exception<Rcs::JointLimitException>(m, "JointLimitException");
    
    // Define BoxSpace as class. It's not providing a constructor here, it's just meant to be passed to python for information
    py::class_<Rcs::BoxSpace>(m, "BoxSpace")
        .def_property_readonly("min", &Rcs::BoxSpace::getMin,
                               py::return_value_policy::reference_internal).def_property_readonly(
        "max", &Rcs::BoxSpace::getMax, py::return_value_policy::reference_internal).def_property_readonly("names", [](
        const Rcs::BoxSpace& thiz) -> py::object {
        auto& names = thiz.getNames();
        if (names.empty()) {
            return py::none();
        }
        return py::cast(names);
    });
    
    // Define simulator base class
    py::class_<Rcs::RcsSimEnv>(m, "RcsSimEnv").def(py::init([](py::kwargs kwargs) {
                                                       // Get properties from xml or python
                                                       Rcs::PropertySource* config;
                                                       std::string configFileName;
                                                       if (try_get(kwargs, "experimentConfigFile", configFileName)) {
                                                           config = new Rcs::PropertySourceXml(configFileName.c_str());
                                                       }
                                                       else {
                                                           config = new Rcs::PropertySourceDict(kwargs);
                                                       }
                                                       // Create config object, takes ownership of property source
                                                       return new Rcs::RcsSimEnv(config);
                                                   })
        )
        .def("step", &Rcs::RcsSimEnv::step, py::arg("action"), py::arg("disturbance") = py::none(),
             py::call_guard<py::gil_scoped_release>())
        .def("reset", [](Rcs::RcsSimEnv& self, py::object domainParam, const MatNd* initState) {
                 Rcs::PropertySource* domainParamSource = Rcs::PropertySource::empty();
                 if (!domainParam.is_none()) {
                     domainParamSource = new Rcs::PropertySourceDict(domainParam);
                 }
                 MatNd* result = self.reset(domainParamSource, initState);
                 if (!domainParam.is_none()) {
                     delete domainParamSource;
                 }
                 return result;
             },
             py::arg("domainParam").none(true) = py::none(), py::arg("initState").none(true) = py::none()
        )
        .def("render", &Rcs::RcsSimEnv::render, py::arg("mode") = "human", py::arg("close") = false)
        .def("toggleVideoRecording", &Rcs::RcsSimEnv::toggleVideoRecording)
        .def("setTransitionNoiseBuffer", &Rcs::RcsSimEnv::setTransitionNoiseBuffer)
        .def("saveConfigXML", [](Rcs::RcsSimEnv& self, const char* fileName) {
            self.getConfig()->properties->saveXML(fileName, "Experiment");
        })
        .def("getBodyPosition",
             [](Rcs::RcsSimEnv& self, const char* bodyName, const char* refBodyName, const char* refFrameName) {
                 const RcsBody* body = RcsGraph_getBodyByName(self.getConfig()->graph, bodyName);
                 RCHECK(body);
                 const RcsBody* refBody = RcsGraph_getBodyByName(self.getConfig()->graph, refBodyName);
                 const RcsBody* refFrame = RcsGraph_getBodyByName(self.getConfig()->graph, refFrameName);
                 double I_r[3];
            
                 // Effector-fixed reference point in world coordinates
                 Vec3d_copy(I_r, body->A_BI->org);
            
                 // Transform to reference frame: ref_r = A_ref-I * (I_r - I_r_refBdy)
                 if (refBody != NULL) {
                     Vec3d_subSelf(I_r, refBody->A_BI->org);     // I_r -= I_r_refBdy
                
                     // refBody and refFrame, but they differ: refFrame_r = A_refFrame-I*I_r
                     if ((refFrame != NULL) && (refFrame != refBody)) {
                         Vec3d_rotateSelf(I_r, refFrame->A_BI->rot);
                     }
                         // refBody and refFrame are the same: refFrame_r = A_refBody-I*I_r
                     else {
                         Vec3d_rotateSelf(I_r, refBody->A_BI->rot);
                     }
                 }
                     // No refBody, but refFrame: Rotate into refFrame coordinates
                 else {
                     // Rotate into refFrame if it exists
                     if (refFrame != NULL) {
                         Vec3d_rotateSelf(I_r, refFrame->A_BI->rot);
                     }
                
                 }
                 MatNd* pos = NULL;
                 pos = MatNd_create(3, 1);
                 for (unsigned int i = 0; i < 3; i++) {
                     MatNd_set(pos, i, 0, I_r[i]);
                 }
                 return pos;
             },
             py::arg("bodyName"), py::arg("refFrameName"), py::arg("refBodyName")
        )
        .def("getBodyExtents", [](Rcs::RcsSimEnv& self, const char* bodyName, const int shapeIdx) {
                 const RcsBody* body = RcsGraph_getBodyByName(self.getConfig()->graph, bodyName);
                 RCHECK(body);
            
                 MatNd* extents = NULL;
                 extents = MatNd_create(3, 1);
                 for (unsigned int i = 0; i < 3; i++) {
                     MatNd_set(extents, i, 0, body->shape[shapeIdx]->extents[i]);
                 }
                 return extents;
             },
             py::arg("bodyName"), py::arg("shapeIdx")
        )
            
            // Properties
        .def_property_readonly("observationSpace", &Rcs::RcsSimEnv::observationSpace)
        .def_property_readonly("actionSpace", &Rcs::RcsSimEnv::actionSpace)
        .def_property_readonly("initStateSpace", &Rcs::RcsSimEnv::initStateSpace)
        .def_property_readonly("domainParam", [](Rcs::RcsSimEnv& self) {
            // Expose the domain parameters to the Python side
            py::dict result;
            Rcs::PropertySourceDict psink(result);
            self.getPhysicsManager()->getValues(&psink);
            return result;
        })
        .def_property_readonly("internalStateDim", &Rcs::RcsSimEnv::getInternalStateDim)
        .def_property_readonly("dt", [](Rcs::RcsSimEnv& self) {
            return self.getConfig()->dt;
        })
        .def_property_readonly("lastAction", &Rcs::RcsSimEnv::getCurrentAction, py::return_value_policy::copy)
        .def_property_readonly("lastObservation", &Rcs::RcsSimEnv::getCurrentObservation,
                               py::return_value_policy::copy);
    
    // Define ControlPolicy and MLPPolicy for tests
    py::class_<Rcs::ControlPolicy> controlPolicy(m, "ControlPolicy");
    controlPolicy.def(py::init<Rcs::ControlPolicy* (*)(const char*, const char*)>(&Rcs::ControlPolicy::create));
    controlPolicy.def("__call__", [](Rcs::ControlPolicy& self, const MatNd* input, unsigned int output_size) {
        MatNd* output = MatNd_create(output_size, 1);
        self.computeAction(output, input);
        return output;
    });
    controlPolicy.def("reset", &Rcs::ControlPolicy::reset);
    controlPolicy.def_property_readonly_static("types", [](py::handle /* self */) {
        return Rcs::ControlPolicy::getTypeNames();
    });
    
    // Due to the way this class works, we don't use it in practice
    py::class_<Rcs::MLPPolicy>(m, "MLPPolicy", controlPolicy)
        .def(py::init(&Rcs::loadMLPPolicyFromXml));
    
    // Define gui stuff if available
//#ifdef GUI_AVAILABLE
//    define_gui_classes(m);
//#endif

    m.def("saveExperimentParams", [](py::dict& config, const char* filename){
        std::unique_ptr<Rcs::PropertySource> ps(new Rcs::PropertySourceDict(config));
        ps->saveXML(filename, "Experiment");
    }, py::arg("config"), py::arg("filename"));

    // Define some utility functions for interacting with RCS
    
    // Sets the rcs log level
    m.def("setLogLevel", [](int level) { RcsLogLevel = level; });
    
    // Adds a directory to the resource path
    m.def("addResourcePath", [](const char* path) { return Rcs_addResourcePath(path); });
    
    // Check if physics engine is available (can't list, unfortunately)
    m.def("supportsPhysicsEngine", &Rcs::PhysicsFactory::hasEngine);
    
    // Control vortex log level, setting it to warnings only by default (this avoids log spam)
    Rcs::setVortexLogLevel("warn");
    // Allow changing it if desired
    m.def("setVortexLogLevel", &Rcs::setVortexLogLevel);
}




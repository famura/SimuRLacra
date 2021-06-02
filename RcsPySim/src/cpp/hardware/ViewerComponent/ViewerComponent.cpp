#include "ViewerComponent.h"

#include <RcsViewer.h>
#include <GraphNode.h>
#include <FTSensorNode.h>
#include <HUD.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_timer.h>


namespace Rcs
{

class LockedFTS : public Rcs::FTSensorNode
{
public:
    
    LockedFTS(const RcsSensor* fts, pthread_mutex_t* mtx) : FTSensorNode(fts), graphMtx(mtx)
    {
    }
    
    bool frameCallback()
    {
        pthread_mutex_lock(graphMtx);
        FTSensorNode::frameCallback();
        pthread_mutex_unlock(graphMtx);
        
        return false;
    }
    
    pthread_mutex_t* graphMtx;
};


class ComponentViewer : public Rcs::Viewer
{
public:
    
    ComponentViewer(const RcsGraph* desired, const RcsGraph* current) :
        Rcs::Viewer(), gDes(NULL), gCurr(NULL), hud(NULL)
    {
        pthread_mutex_init(&this->graphMtx, NULL);
        
        if (desired != NULL) {
            this->gDes = RcsGraph_clone(desired);
            
            this->gnDes = new Rcs::GraphNode(this->gDes);
            gnDes->setGhostMode(true, "RED");
            add(gnDes.get());
        }
        
        if (current != NULL) {
            this->gCurr = RcsGraph_clone(current);
            this->gnCurr = new Rcs::GraphNode(this->gCurr);
            add(this->gnCurr);
            
            RCSGRAPH_TRAVERSE_SENSORS(this->gCurr) {
                if (SENSOR->type == RCSSENSOR_LOAD_CELL) {
                    osg::ref_ptr<LockedFTS> ftn = new LockedFTS(SENSOR, &this->graphMtx);
                    add(ftn.get());
                }
            }
        }
        
        this->hud = new Rcs::HUD();
        add(hud.get());
    }
    
    ~ComponentViewer()
    {
        // Graph destruction checks for NULL pointers
        RcsGraph_destroy(this->gDes);
        RcsGraph_destroy(this->gCurr);
        
        pthread_mutex_destroy(&this->graphMtx);
    }
    
    void frame()
    {
        
        if (isInitialized() == false) {
            init();
        }
        
        pthread_mutex_lock(&this->graphMtx);
        
        if (this->gDes != NULL) {
            RcsGraph_setState(this->gDes, NULL, NULL);
        }
        
        if (this->gCurr != NULL) {
            RcsGraph_setState(this->gCurr, NULL, NULL);
        }
        
        pthread_mutex_unlock(&this->graphMtx);
        
        double dtFrame = Timer_getSystemTime();
        
        // Publish all queued events before the frame() call
        userEventMtx.lock();
        for (size_t i = 0; i < userEventStack.size(); ++i) {
            getOsgViewer()->getEventQueue()->userEvent(userEventStack[i].get());
        }
        userEventStack.clear();
        userEventMtx.unlock();
        
        lock();
        viewer->frame();
        unlock();
        
        dtFrame = Timer_getSystemTime() - dtFrame;
        this->fps = 0.9*this->fps + 0.1*(1.0/dtFrame);
    }
    
    void setText(const std::string& text)
    {
        lock();
        hud->setText(text);
        unlock();
    }
    
    RcsGraph* gDes;
    RcsGraph* gCurr;
    osg::ref_ptr<HUD> hud;
    osg::ref_ptr<Rcs::GraphNode> gnDes;
    osg::ref_ptr<Rcs::GraphNode> gnCurr;
    pthread_mutex_t graphMtx;
};
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::ViewerComponent::ViewerComponent(
    RcsGraph* graphCurr,
    bool syncWithEventLoop_) :
    Rcs::SensorComponent(), desiredGraph(NULL), currentGraph(graphCurr),
    viewer(NULL), kc(NULL), syncWithEventLoop(syncWithEventLoop_)
{
    init();
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::ViewerComponent::ViewerComponent(
    RcsGraph* graphDes, RcsGraph* graphCurr,
    bool syncWithEventLoop_) :
    Rcs::SensorComponent(), desiredGraph(graphDes), currentGraph(graphCurr),
    viewer(NULL), kc(NULL), syncWithEventLoop(syncWithEventLoop_)
{
    init();
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::ViewerComponent::init()
{
    this->viewer = new Rcs::ComponentViewer(desiredGraph, currentGraph);
    getKeyCatcher();
    
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::ViewerComponent::~ViewerComponent()
{
    delete this->viewer;
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::ViewerComponent::updateGraph(RcsGraph* graph)
{
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::ViewerComponent::postUpdateGraph()
{
    pthread_mutex_lock(&viewer->graphMtx);
    
    if (desiredGraph != NULL) {
        MatNd_copy(viewer->gDes->q, desiredGraph->q);
    }
    
    if (currentGraph != NULL) {
        MatNd_copy(viewer->gCurr->q, currentGraph->q);
        
        RcsSensor* src = currentGraph->sensor;
        RcsSensor* dst = viewer->gCurr->sensor;
        
        while (src != NULL) {
            if (src->type == RCSSENSOR_LOAD_CELL) {
                MatNd_copy(dst->rawData, src->rawData);
            }
            src = src->next;
            dst = dst->next;
        }
        
    }
    
    pthread_mutex_unlock(&viewer->graphMtx);
    
    if (this->syncWithEventLoop == true) {
        viewer->frame();
    }
}

/*******************************************************************************
 *
 ******************************************************************************/
const char* Rcs::ViewerComponent::getName() const
{
    return "ViewerComponent";
}

/*******************************************************************************
 *
 ******************************************************************************/
double Rcs::ViewerComponent::getCallbackUpdatePeriod() const
{
    return 1.0/viewer->updateFrequency();
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::ViewerComponent::setText(const std::string& text)
{
    viewer->setText(text);
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::KeyCatcher* Rcs::ViewerComponent::getKeyCatcher()
{
    if (this->kc == NULL) {
        this->kc = new Rcs::KeyCatcher();
        viewer->add(kc);
    }
    
    return this->kc;
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::Viewer* Rcs::ViewerComponent::getViewer()
{
    return this->viewer;
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::BodyNode* Rcs::ViewerComponent::getBodyNodePtrFromDesiredGraph(const char* name)
{
    return viewer->gnDes->getBodyNode(name);
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::BodyNode* Rcs::ViewerComponent::getBodyNodePtrFromCurrentGraph(const char* name)
{
    return viewer->gnCurr->getBodyNode(name);
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::ViewerComponent::lock()
{
    viewer->lock();
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::ViewerComponent::unlock()
{
    viewer->unlock();
}

bool Rcs::ViewerComponent::startThread()
{
    if (this->syncWithEventLoop == false) {
        viewer->runInThread();
    }
    // do nothing
    return false;
}

bool Rcs::ViewerComponent::stopThread()
{
    // do nothing
    return false;
}

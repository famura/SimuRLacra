#ifndef RCS_VIEWERCOMPONENT_H
#define RCS_VIEWERCOMPONENT_H


#include "SensorComponent.h"

#include <RcsViewer.h>
#include <KeyCatcher.h>
#include <BodyNode.h>


namespace Rcs
{
class ComponentViewer;

class ViewerComponent : public SensorComponent
{
public:
    ViewerComponent(RcsGraph* currentGraph, bool syncWithEventLoop = false);
    
    ViewerComponent(
        RcsGraph* desiredGraph, RcsGraph* currentGraph,
        bool syncWithEventLoop = false);
    
    virtual ~ViewerComponent();
    
    void updateGraph(RcsGraph* graph);
    
    void postUpdateGraph();
    
    const char* getName() const;
    
    double getCallbackUpdatePeriod() const;
    
    void setText(const std::string& text);
    
    KeyCatcher* getKeyCatcher();
    
    Viewer* getViewer();
    
    BodyNode* getBodyNodePtrFromDesiredGraph(const char* name);
    
    BodyNode* getBodyNodePtrFromCurrentGraph(const char* name);
    
    void lock();
    
    void unlock();
    
    bool startThread();
    
    bool stopThread();

private:
    void init();
    
    const RcsGraph* desiredGraph;
    const RcsGraph* currentGraph;
    ComponentViewer* viewer;
    KeyCatcher* kc;
    bool syncWithEventLoop;
    
    ViewerComponent(const ViewerComponent&);
    
    ViewerComponent& operator=(const ViewerComponent&);
};

}

#endif   // RCS_VIEWERCOMPONENT_H

/*
register all plugins
*/
#pragma once
#include "macro.h"
//#include "logger.h"
#include <maya/MStatus.h>
#include <maya/MFnPlugin.h>
#include <maya/MObject.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MGlobal.h>
#include <maya/MPxNode.h>
////#include <maya/MPxTransform.h>
//#include <maya/MDrawRegistry.h>
//
//
////#include "mayaNode/strataGraphNode.h"
//#include "mayaNode/strataOpNodeBase.h"
//#include "mayaNode/strataPointNode.h"
//#include "mayaNode/strataElementOpNode.h"
//#include "mayaNode/strataShapeNode.h"
//#include "mayaNode/strataShapeGeometryOverride.h"
//
//#include "mayaNode/matrixCurve.h"
//#include "mayaNode/frameCurve.h"

//#include "mayanode/stratacurvenode.h"
//#include "mayanode/stratasurfacenode.h"

const char* kAUTHOR = "ed";
const char* kVERSION = "1.0";
const char* kREQUIRED_API_VERSION = "Any";


#define REGISTER_NODE(NODE) \
    s = fnPlugin.registerNode( \
        NODE::kNODE_NAME, \
        NODE::kNODE_ID, \
        NODE::creator, \
        NODE::initialize \
    ); \
    MCHECK(s, "cannot register node " + NODE::kNODE_NAME); \
    /*CHECK_MSTATUS_AND_RETURN_IT(s); \*/
//pluginObjects.insert(NODE);\

#define REGISTER_DEFORMER(NODE) \
    s = fnPlugin.registerNode( \
        NODE::kNODE_NAME, \
        NODE::kNODE_ID, \
        NODE::creator, \
        NODE::initialize, \
        MPxNode::Type::kDeformerNode \
    ); \
    MCHECK(s, "cannot register deformer " + NODE::kNODE_NAME); \


#define DEREGISTER_NODE(NODE) \
    s = fnPlugin.deregisterNode( \
        NODE::kNODE_ID ); \
    MCHECK(s, "failed to deregister " + NODE::kNODE_NAME); \

#define REGISTER_NODE_TYPE(NODE, NODE_TYPE) \
    s = fnPlugin.registerNode( \
        NODE::kNODE_NAME, \
        NODE::kNODE_ID, \
        NODE::creator, \
        NODE::initialize, \
        NODE_TYPE \
    ); \
    MCHECK(s, "failed to register node type " + NODE::kNODE_NAME); \
 // MPxNode::kDependNode

/*
macros shamelessly lifted from yantor3d
thanks mate
*/
//


static MString sCustomSpriteShaderRegistrantId("customSpriteShaderRegistrantId");
static MString sCustomSpriteShaderDrawdbClassification("drawdb/shader/surface/customSpriteShader");

static const MString svp2BlinnShaderRegistrantId("vp2BlinnShaderRegistrantId");
static const MString GLRegistrandId("GLLocatorNodePlugin");

/* strata draw DB strings*/
static const MString sDrawDbClassification("drawdb/geometry/strataShapeNodeOverride");
static const MString sDrawRegistrantId("strataShapeNodeOverridePlugin");

MStatus initializePlugin(MObject obj) {

    //DEBUGS("")
    //    DEBUGS("")
    //    DEBUGS("")
    //    DEBUGS("_")
    //    DEBUGS("_")
    //    DEBUGS("___")
        //LOG("initialising strata")
        MFnPlugin fnPlugin(obj, kAUTHOR, kVERSION, kREQUIRED_API_VERSION);
    MStatus s = MStatus::kSuccess;


    DEBUGS("initialised wpplugin");
    return s;
}

MStatus uninitializePlugin(MObject obj) {

    MStatus s;
    s = MS::kSuccess;
    //LOG("uninitialising wpplugin");

    MFnPlugin fnPlugin(obj);

    DEBUGS("uninitialised wpplugin");
    return s;

}

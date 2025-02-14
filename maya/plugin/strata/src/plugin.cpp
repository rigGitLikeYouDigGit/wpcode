/*
register all plugins
*/
#pragma once
#include "macro.h"

#include <maya/MFnPlugin.h>
#include <maya/MObject.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MGlobal.h>
#include <maya/MPxNode.h>
#include <maya/MPxTransform.h>
#include <maya/MDrawRegistry.h>

#include "stratapoint.h"
#include "stratacurve.h"

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
    MCHECK(s, "cannot register node " << NODE::kNODE_NAME); \
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
    MCHECK(s, "cannot register deformer " << NODE::kNODE_NAME); \


#define DEREGISTER_NODE(NODE) \
    s = fnPlugin.deregisterNode( \
        NODE::kNODE_ID ); \
    MCHECK(s, "failed to deregister " << NODE::kNODE_NAME); \

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

MStatus initializePlugin( MObject obj ){

    DEBUGS("initialising strata")
    MFnPlugin fnPlugin( obj, kAUTHOR, kVERSION, kREQUIRED_API_VERSION);
    MStatus s = MStatus::kSuccess;

    //// strataPoint node
    s = fnPlugin.registerTransform(
        StrataPoint::kNODE_NAME,
        StrataPoint::kNODE_ID,
        StrataPoint::creator,
        StrataPoint::initialize,
        StrataPointMatrix::creator,
        StrataPointMatrix::id,
        &StrataPoint::drawDbClassification
    );
    if (MS::kSuccess != s) {
        cerr << 82 << "failed to register node type " + StrataPoint::kNODE_NAME; 
        return MS::kFailure;
    };
    s = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
        StrataPoint::drawDbClassification,
        StrataPoint::drawRegistrantId,
        StrataPointDrawOverride::creator);
    if (!s) {
        s.perror("registerDrawOverrideCreator");
        return s;
    }

    //// strataCurve node
    s = fnPlugin.registerTransform(
        StrataCurve::kNODE_NAME,
        StrataCurve::kNODE_ID,
        StrataCurve::creator,
        StrataCurve::initialize,
        StrataCurveMatrix::creator,
        StrataCurveMatrix::id
        //&StrataPoint::drawDbClassification
    );
    if (MS::kSuccess != s) {
        cerr << 82 << "failed to register node type " + StrataCurve::kNODE_NAME;
        return MS::kFailure;
    };


    return s;
}

MStatus uninitializePlugin( MObject obj ){

    MStatus s;
    s = MS::kSuccess;
    MFnPlugin fnPlugin(obj);

    s = MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
        StrataPoint::drawDbClassification,
        StrataPoint::drawRegistrantId);
    if (!s) {
        s.perror("deregisterDrawOverrideCreator");
        return s;
    }

    DEREGISTER_NODE(StrataPoint);
    DEREGISTER_NODE(StrataCurve);
    DEBUGS("uninitialised strata")
    return s;

}

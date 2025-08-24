/*
register all plugins
*/
#pragma once
#include "macro.h"
#include "logger.h"

#include <maya/MFnPlugin.h>
#include <maya/MObject.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MGlobal.h>
#include <maya/MPxNode.h>
//#include <maya/MPxTransform.h>
#include <maya/MDrawRegistry.h>


//#include "mayaNode/strataGraphNode.h"
#include "mayaNode/strataOpNodeBase.h"
#include "mayaNode/strataPointNode.h"
#include "mayaNode/strataElementOpNode.h"
#include "mayaNode/strataShapeNode.h"
#include "mayaNode/strataShapeGeometryOverride.h"

#include "mayaNode/matrixCurve.h"
#include "mayaNode/frameCurve.h"

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

MStatus initializePlugin( MObject obj ){

    DEBUGS("")
    DEBUGS("")
    DEBUGS("")
    DEBUGS("_")
    DEBUGS("_")
    DEBUGS("___")
    LOG("initialising strata")
    MFnPlugin fnPlugin( obj, kAUTHOR, kVERSION, kREQUIRED_API_VERSION);
    MStatus s = MStatus::kSuccess;

    //REGISTER_NODE(StrataGraphNode);
    /*if (MS::kSuccess != s) {
        cerr << 82 << "failed to register node type " + StrataGraphNode::kNODE_NAME;
        return MS::kFailure;
    };*/

    //REGISTER_NODE(StrataOpNodeBase);



    ////// strataPoint node
    s = fnPlugin.registerNode(
        StrataPointNode::kNODE_NAME,
        StrataPointNode::kNODE_ID,
        StrataPointNode::creator,
        StrataPointNode::initialize,
        MPxNode::kLocatorNode,
        &StrataPointNode::drawDbClassification
    );
    if (MS::kSuccess != s) {
        cerr << 82 << "failed to register node type " + StrataPointNode::kNODE_NAME;
        return MS::kFailure;
    };    
    s = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
        StrataPointNode::drawDbClassification,
        StrataPointNode::drawRegistrantId,
        StrataPointNodeDrawOverride::creator);
    if (!s) {
        s.perror("could not register drawOverrideCreator for StrataPoint");
        return s;
    }

    REGISTER_NODE(StrataElementOpNode);
    s = fnPlugin.registerNode(StrataElementOpNode::kNODE_NAME, StrataElementOpNode::kNODE_ID, StrataElementOpNode::creator, StrataElementOpNode::initialize); if (MS::kSuccess != s) {
        cerr << 121 << "cannot register node " << StrataElementOpNode::kNODE_NAME; return MS::kFailure;
    };
    

    /////// SHAPE NODE /////////
    s = fnPlugin.registerShape(
        StrataShapeNode::kNODE_NAME,
        StrataShapeNode::kNODE_ID,
        &StrataShapeNode::creator,
        &StrataShapeNode::initialize,
        StrataShapeUI::creator,
        //nullptr, // apparently don't need the dumb legacy creator after all
        &StrataShapeNode::drawDbClassification
    );
    MCHECK(s, "ERROR registering Strata shape node");

    if (!s)
    {
        std::cerr << "Failed to register geometryOverrideExample2_shape." << std::endl;
        return s;
    }

    s = MHWRender::MDrawRegistry::registerGeometryOverrideCreator(
        //sDrawDbClassification,
        StrataShapeNode::drawDbClassification,
        sDrawRegistrantId,
        StrataShapeGeometryOverride::Creator
    );
    MCHECK(s, "ERROR registering Strata shape geometry override");

    if (!s)
    {
        std::cerr << "Failed to register Viewport 2.0 geometry override." << std::endl;
        return s;
    }

    /// bonus matrix curve node
    REGISTER_NODE(MatrixCurveNode);

    REGISTER_NODE(FrameCurveNode);

    l("initialised Strata");
    return s;
}

MStatus uninitializePlugin( MObject obj ){

    MStatus s;
    s = MS::kSuccess;
    LOG("uninitialising strata");

    MFnPlugin fnPlugin(obj);

    s = MHWRender::MDrawRegistry::deregisterGeometryOverrideCreator(
       // sDrawDbClassification, 
        StrataShapeNode::drawDbClassification,
        sDrawRegistrantId);
    MCHECK(s, "could not deregister drawOverrideCreator for Strata shape");

    s = MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
        StrataPointNode::drawDbClassification,
        StrataPointNode::drawRegistrantId);
    MCHECK(s, "could not deregister drawOverride for StrataPoint");


    DEREGISTER_NODE(StrataPointNode);

    DEREGISTER_NODE(StrataElementOpNode);

    DEREGISTER_NODE(StrataShapeNode);

    DEREGISTER_NODE(MatrixCurveNode);

    DEREGISTER_NODE(FrameCurveNode);

    l("uninitialised strata");
    return s;

}

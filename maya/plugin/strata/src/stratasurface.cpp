
#pragma once
#include <vector>
//#include <array>
#include <maya/MMatrix.h>
#include <maya/MDrawRegistry.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MPointArray.h>
#include <maya/MGlobal.h>
#include <maya/MEventMessage.h>
#include <maya/MDGModifier.h>
#include <maya/MString.h>

#include <maya/MFnDependencyNode.h>
#include <maya/MFnTransform.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnGenericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnNurbsCurve.h>


#include "macro.h"
#include "api.h"
#include "stratasurface.h"
#include "lib.cpp"


using namespace ed;

MTypeId StrataSurfaceMatrix::id(0x00122C2A);


MTypeId StrataSurface::kNODE_ID(0x00122C2A);
MString StrataSurface::kNODE_NAME("StrataSurface");

MString StrataSurface::drawDbClassification("drawdb/geometry/StrataSurface");
MString StrataSurface::drawRegistrantId("StrataSurfacePlugin");


MObject StrataSurface::aBalanceWheel;
MObject StrataSurface::aStOutMesh;

MObject StrataSurface::aStEditMode;
MObject StrataSurface::aStUiData;

void StrataSurface::postConstructor() {
    // create a nurbsCurve shape node under this transform, 
    // connect this node's outCurve attribute to it.
    // also do the upcurve?
    // nah, naming would get inconsistent, and upcurve not always needed
    //MStatus s(MS::kSuccess);

    //MFnDagNode meshFn;
    //MFnDagNode thisFn(thisMObject());
    //MDGModifier dgMod;
    //MObject meshObj = meshFn.create(
    //    "mesh", thisFn.name() + "Shape",
    //    thisMObject());
    //meshFn.setObject(meshObj);


    //const MString outPlugName("outMesh");
    //MPlug meshOutPlug = thisFn.findPlug(outPlugName, true, &s);
    //MPlug meshCreatePlug = meshFn.findPlug(MString("inMesh"), true, &s);

    //dgMod.connect(meshOutPlug, meshCreatePlug);
    //dgMod.doIt();
}

MStatus StrataSurface::initialize() {
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;

    //inputs
   

    // outputs
    aBalanceWheel = nFn.create("balanceWheel", "balanceWheel",
        MFnNumericData::kBoolean, 0);
    nFn.setWritable(false);
    nFn.setCached(false);

    aStOutMesh = tFn.create("stOutMesh", "stOutMesh",
        MFnData::kMesh);
    tFn.setWritable(false);
    tFn.setCached(true);

    aStEditMode = nFn.create("stEditMode", "stEditMode", MFnNumericData::kBoolean, 0);
    nFn.setChannelBox(1);
    aStUiData = tFn.create("stUiData", "stUiData", MFnData::kString);


    std::vector<MObject> driverObjs = {
        aStEditMode, aStUiData,
 
    };

    std::vector<MObject>drivenObjs = {
        aBalanceWheel,
        aStOutMesh,
    };

    //addAttributes<StrataSurface>(driverObjs);
    //addAttributes<StrataSurface>(drivenObjs);

    setAttributesAffect<StrataSurface>(driverObjs, drivenObjs);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}


MStatus StrataSurface::compute(const MPlug& plug, MDataBlock& data) {
    MStatus s = MS::kSuccess;

    if (data.isClean(plug)) {
        return s;
    }
    DEBUGS("surface compute");

    // flip balancewheel
    data.outputValue(aBalanceWheel).setBool(!data.outputValue(aBalanceWheel).asBool());
    data.setClean(aBalanceWheel);

    data.setClean(plug);
    return s;
}

StrataSurfaceMatrix::StrataSurfaceMatrix(const MMatrix& mat) : MPxTransformationMatrix(mat) {
}


//MStatus StrataSurface::computeLocalTransformation(MPxTransformationMatrix* xform, MDataBlock& data) {
//    /* rockingTransform example in the dev kit delegates more functionality to the custom
//    behaviour of the transformation matrix itself, but it seems needlessly complex to me -
//    here we just layer the local offsets on top of each other
//    */
//    MS s = MS::kSuccess;
//    s = MPxTransform::computeLocalTransformation(xform, data);
//    CHECK_MSTATUS(s);
//    // insert the two custom matrices before the vanilla behaviour - 
//    // HOPEFULLY this lets them come after the normal dag node parent transformation,
//    // but before the normal TRS attributes are combined
//    MMatrix finalParentMat = data.outputValue(aStFinalDriverOutMatrix).asMatrix();
//    MMatrix finalLocalMat = data.outputValue(aStFinalLocalOffsetMatrix).asMatrix();
//    const MMatrix endMat = finalParentMat * finalLocalMat * xform->asMatrix();
//    *xform = endMat;
//    DEBUGS("computed local transformation")
//        return s;
//
//}

//MHWRender::DrawAPI StrataSurfaceDrawOverride::supportedDrawAPIs() const
//{
//    // this plugin supports both GL and DX
//    return (MHWRender::kOpenGL | MHWRender::kDirectX11 | MHWRender::kOpenGLCoreProfile);
//}
//bool StrataSurfaceDrawOverride::isBounded(const MDagPath& /*objPath*/,
//    const MDagPath& /*cameraPath*/) const
//{
//    return true;
//}
//
//float StrataSurfaceDrawOverride::getMultiplier(const MDagPath& objPath) const
//{
//    // Retrieve value of the size attribute from the node
//    // don't bother converting to distance and back, units are units
//    MStatus status;
//    MObject footprintNode = objPath.node(&status);
//    if (status)
//    {
//        MPlug plug(footprintNode, StrataSurface::aStRadius);
//        if (!plug.isNull())
//        {
//            return plug.asFloat();
//        }
//    }
//    return 1.0f;
//}
//
//MBoundingBox StrataSurfaceDrawOverride::boundingBox(
//    const MDagPath& objPath,
//    const MDagPath& cameraPath) const
//{
//    MPoint corner(-1, -1, -1);
//    float multiplier = getMultiplier(objPath);
//    return MBoundingBox(corner * multiplier, corner * multiplier * -1.0);
//}
//
//
//// Called by Maya each time the object needs to be drawn.
//MUserData* StrataSurfaceDrawOverride::prepareForDraw(
//    const MDagPath& objPath,
//    const MDagPath& cameraPath,
//    const MHWRender::MFrameContext& frameContext,
//    MUserData* oldData)
//{
//    // Any data needed from the Maya dependency graph must be retrieved and cached in this stage.
//    // There is one cache data for each drawable instance, if it is not desirable to allow Maya to handle data
//    // caching, simply return null in this method and ignore user data parameter in draw callback method.
//    // e.g. in this sample, we compute and cache the data for usage later when we create the 
//    // MUIDrawManager to draw footprint in method addUIDrawables().
//    //DEBUGS("prepare for draw")
//    StrataSurfaceUserData* data = dynamic_cast<StrataSurfaceUserData*>(oldData);
//    if (!data)
//    {
//        data = new StrataSurfaceUserData();
//    }
//    MStatus s = MS::kSuccess;
//
//    float fMultiplier = getMultiplier(objPath);
//
//    // get correct color and depth priority based on the state of object, e.g. active or dormant
//    MObject node = objPath.node();
//    data->fColor = MHWRender::MGeometryUtilities::wireframeColor(objPath);
//    MVector worldPos = MFnTransform(objPath).getTranslation(MSpace::kWorld, &s);
//    CHECK_MSTATUS(s);
//    data->pos = worldPos;
//
//    if (!node.isNull()) {
//
//        MPlug rgbPlug = MFnDependencyNode(node).findPlug("objectColorRGB", s);
//        CHECK_MSTATUS(s);
//        if (!rgbPlug.isNull()) {
//            //DEBUGS("getting rgb colour");
//            // get RGB colour
//            MColor rgb = MColor(
//                rgbPlug.child(0).asFloat(),
//                rgbPlug.child(1).asFloat(),
//                rgbPlug.child(2).asFloat()
//            );
//            //DEBUGS(rgbPlug.child(0).name());
//            data->fColor = rgb;
//        }
//        // fall back to positions if colour not set
//        if (data->fColor == MColor::kOpaqueBlack) {
//            worldPos = (worldPos.normal() + MVector(1.0, 1.0, 1.0)) / 2.0;
//            data->fColor.set(MColor::kRGB,
//                float(worldPos[0]), float(worldPos[1]), float(worldPos[2]));
//        }
//
//        MPlug radiusPlug = MFnDependencyNode(node).findPlug(StrataSurface::aStRadius, s);
//        CHECK_MSTATUS(s);
//        data->size = radiusPlug.asFloat();
//    }
//
//    switch (MHWRender::MGeometryUtilities::displayStatus(objPath))
//    {
//    case MHWRender::kLead:
//        data->fColor = data->fColor * 2.0;
//        data->fDepthPriority = MHWRender::MRenderItem::sActiveWireDepthPriority;
//        break;
//    case MHWRender::kActive:
//        data->fColor = data->fColor * 1.5;
//        data->fDepthPriority = MHWRender::MRenderItem::sActiveWireDepthPriority;
//        break;
//    case MHWRender::kHilite:
//        data->fColor = data->fColor * 1.5;
//        data->fDepthPriority = MHWRender::MRenderItem::sActiveWireDepthPriority;
//        break;
//        //case MHWRender::kActiveComponent:
//
//    default:
//        data->fDepthPriority = MHWRender::MRenderItem::sDormantFilledDepthPriority;
//        break;
//    }
//    return data;
//}
//// addUIDrawables() provides access to the MUIDrawManager, which can be used
//// to queue up operations for drawing simple UI elements such as lines, circles and
//// text. To enable addUIDrawables(), override hasUIDrawables() and make it return true.
//void StrataSurfaceDrawOverride::addUIDrawables(
//    const MDagPath& objPath,
//    MHWRender::MUIDrawManager& drawManager,
//    const MHWRender::MFrameContext& frameContext,
//    const MUserData* data)
//{
//    // Get data cached by prepareForDraw() for each drawable instance, then MUIDrawManager 
//    // can draw simple UI by these data.
//    //DEBUGS("addUIDrawables");
//    StrataSurfaceUserData* pLocatorData = (StrataSurfaceUserData*)data;
//    if (!pLocatorData)
//    {
//        DEBUGS("no prev user data")
//            return;
//    }
//    // just draw a circle
//    drawManager.beginDrawable();
//    drawManager.setColor(pLocatorData->fColor);
//    drawManager.setDepthPriority(pLocatorData->fDepthPriority);
//    drawManager.setLineWidth(1.0);
//    MPoint pos(0.0, 0.0, 0.0); // Position of the sphere
//    if (frameContext.getDisplayStyle() & MHWRender::MFrameContext::kGouraudShaded) {
//        drawManager.sphere(pos, pLocatorData->size, true);
//    }
//    else {
//        MDoubleArray dir = frameContext.getTuple(MFrameContext::kViewDirection);
//        MVector normal(dir[0], dir[1], dir[2]);
//        drawManager.circle(pos, normal, pLocatorData->size, false);
//    }
//    drawManager.endDrawable();
//}



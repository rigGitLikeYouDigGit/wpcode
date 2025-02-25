
#pragma once
#include <vector>
//#include <array>



#include "../macro.h"
#include "../api.h"
#include "stratapointnode.h"
#include "../lib.cpp"


using namespace ed;



MTypeId StrataPoint::kNODE_ID(0x00122C1C);
MString StrataPoint::kNODE_NAME("strataPoint");

MString StrataPoint::drawDbClassification("drawdb/geometry/stratapoint");
MString StrataPoint::drawRegistrantId("StrataPointPlugin");

//MObject MPxTransform::offsetParentMatrix;

MObject StrataPoint::aStDriver;
MObject StrataPoint::aStDriverType;
MObject StrataPoint::aStDriverWeight;

MObject StrataPoint::aStDriverClosestPoint;
MObject StrataPoint::aStDriverUseClosestPoint;

MObject StrataPoint::aStDriverPointMatrix;

MObject StrataPoint::aStDriverCurve;
MObject StrataPoint::aStDriverUpCurve;
MObject StrataPoint::aStDriverRefLengthCurve;
MObject StrataPoint::aStDriverCurveLength;
MObject StrataPoint::aStDriverCurveParam;
MObject StrataPoint::aStDriverCurveLengthParamBlend;
MObject StrataPoint::aStDriverCurveReverseBlend;
MObject StrataPoint::aStDriverCurveNormLengthBlend;

MObject StrataPoint::aStDriverSurface;

MObject StrataPoint::aStDriverNormalizedWeight;
MObject StrataPoint::aStDriverLocalOffsetMatrix;
MObject StrataPoint::aStDriverOutMatrix;
MObject StrataPoint::aStDriverUpdateParamsInEditMode;

MObject StrataPoint::aStEditMode;
MObject StrataPoint::aStFinalDriverOutMatrix;
MObject StrataPoint::aStFinalLocalOffsetMatrix;
MObject StrataPoint::aStFinalOutMatrix;

 MObject StrataPoint::aStName;
 MObject StrataPoint::aStLinkNameToNode;
 MObject StrataPoint::aBalanceWheel;

MObject StrataPoint::aStRadius;
MObject StrataPoint::aStUiData;



MStatus StrataPoint::initialize() {
	MStatus s = MS::kSuccess;
	MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;

    // driver array
    aStDriver = cFn.create("stDriver", "stDriver");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);

    aStDriverType = makeEnumAttr<STDriverType>(
        "stDriverType", STDriverType::point, eFn);
    //eFn.setObject(aStDriverType);
    cFn.addChild(aStDriverType);
    
    aStDriverWeight = nFn.create("stDriverWeight", "stDriverWeight", MFnNumericData::kFloat, 1.0);
    cFn.addChild(aStDriverWeight);

    aStDriverClosestPoint = nFn.create("stDriverClosestPoint", "stDriverClosestPoint",
        MFnNumericData::k3Float);
    cFn.addChild(aStDriverClosestPoint);
    aStDriverUseClosestPoint = nFn.create("stDriverUseClosestPoint", "stDriverUseClosestPoint", MFnNumericData::kBoolean, 0);
    cFn.addChild(aStDriverUseClosestPoint);

    // point driver
    aStDriverPointMatrix = mFn.create("stDriverPointMatrix", "stDriverPointMatrix", MFnMatrixAttribute::kDouble);
    mFn.setDefault(MMatrix());
    cFn.addChild(aStDriverPointMatrix);

    // curve driver
    aStDriverCurve = tFn.create("stDriverCurve", "stDriverCurve", MFnData::kNurbsCurve);
    cFn.addChild(aStDriverCurve);
    aStDriverUpCurve = tFn.create("stDriverUpCurve", "stDriverUpCurve", MFnData::kNurbsCurve);
    cFn.addChild(aStDriverUpCurve);
    aStDriverRefLengthCurve = tFn.create("stDriverRefLengthCurve", "stDriverRefLengthCurve", MFnData::kNurbsCurve);
    cFn.addChild(aStDriverRefLengthCurve);
    aStDriverCurveLength = nFn.create("stDriverCurveLength", "stDriverCurveLength", MFnNumericData::kFloat, 0.0);
    cFn.addChild(aStDriverCurveLength);
    aStDriverCurveParam = nFn.create("stDriverCurveParam", "stDriverCurveParam", MFnNumericData::kFloat, 0.0);
    cFn.addChild(aStDriverCurveParam);
    aStDriverCurveLengthParamBlend = nFn.create("stDriverCurveLengthParamBlend", "stDriverCurveLengthParamBlend", MFnNumericData::kFloat, 0.0);
    cFn.addChild(aStDriverCurveLengthParamBlend);
    aStDriverCurveReverseBlend = nFn.create("stDriverCurveReverseBlend", "stDriverCurveReverseBlend", MFnNumericData::kFloat, 0.0);
    cFn.addChild(aStDriverCurveReverseBlend);

    // surface driver
    aStDriverSurface = gFn.create("stDriverSurface", "stDriverSurface");
    cFn.addChild(aStDriverSurface);

    // final parent matrix for this driver
    aStDriverOutMatrix = mFn.create("stDriverOutMatrix", "stDriverOutMatrix", MFnMatrixAttribute::kDouble);
    mFn.setDefault(MMatrix());
    cFn.addChild(aStDriverOutMatrix);

    // final normalized weight for driver (just convenient)
    aStDriverNormalizedWeight = nFn.create("stDriverNormalizedWeight", "stDriverNormalizedWeight",
        MFnNumericData::kFloat, 0.0);
    nFn.setWritable(false);
    cFn.addChild(aStDriverLocalOffsetMatrix);
    // final offset matrix for this driver
    aStDriverLocalOffsetMatrix = mFn.create("stDriverLocalOffsetMatrix", "stDriverLocalOffsetMatrix", MFnMatrixAttribute::kDouble);
    mFn.setDefault(MMatrix());
    cFn.addChild(aStDriverLocalOffsetMatrix);
    // should driver params update to best fit in edit mode, or only matrix
    aStDriverUpdateParamsInEditMode = nFn.create("stDriverUpdateParamsInEditMode", "stDriverUpdateParamsInEditMode",
        MFnNumericData::kBoolean, 0);
    cFn.addChild(aStDriverUpdateParamsInEditMode);

    // final weighted driver matrix for point
    aStFinalDriverOutMatrix = mFn.create("stFinalDriverMatrix", "stFinalDriverMatrix", MFnMatrixAttribute::kDouble);
    mFn.setWritable(false);
    mFn.setDefault(MMatrix());
    // final local offset from weighted matrix
    aStFinalLocalOffsetMatrix = mFn.create("stFinalLocalMatrix", "stFinalLocalMatrix", MFnMatrixAttribute::kDouble);
    mFn.setWritable(false);
    mFn.setDefault(MMatrix());

    aStFinalOutMatrix = mFn.create("stFinalOutMatrix", "stFinalOutMatrix", MFnMatrixAttribute::kDouble);
    mFn.setWritable(false);
    mFn.setDefault(MMatrix());

    aStEditMode = nFn.create("stEditMode", "stEditMode", MFnNumericData::kBoolean, 0);
	nFn.setChannelBox(1);
	nFn.setKeyable(0);
    nFn.setAffectsWorldSpace(true);

	// radius only affects visuals for now
    aStRadius = nFn.create("stRadius", "stRadius", MFnNumericData::kFloat, 0.1);
	nFn.setAffectsAppearance(true);
    nFn.setAffectsWorldSpace(true);
    nFn.setMin(0.0);
	nFn.setChannelBox(1);

    // semantic unique name for point in strata
    aStName = tFn.create("stName", "stName", MFnData::kString);
    // should the semantic name be linked to the name of the node in maya?
    aStLinkNameToNode = nFn.create("stLinkNameToNode", "stLinkNameToNode", MFnNumericData::kBoolean, true);
    nFn.setChannelBox(1);
    nFn.setKeyable(false);

    // balancewheel for circumventing DG
    aBalanceWheel = nFn.create("balanceWheel", "balanceWheel", MFnNumericData::kBoolean, true);
    nFn.setChannelBox(1);
    nFn.setKeyable(false);

    aStUiData = tFn.create("stUiData", "stUiData", MFnData::kString);

    std::vector<MObject>driverChildObjs = {
        aStDriverType, aStDriverWeight, aStDriverClosestPoint, aStDriverUseClosestPoint,
        aStDriverPointMatrix,
        aStDriverCurve, aStDriverUpCurve, aStDriverRefLengthCurve,
        aStDriverCurveLength, aStDriverCurveParam, aStDriverCurveLengthParamBlend, aStDriverCurveReverseBlend,
        aStDriverSurface, 
        aStDriverLocalOffsetMatrix, aStDriverUpdateParamsInEditMode
    };

    std::vector<MObject>drivenObjs = {
        aStFinalDriverOutMatrix, aStFinalLocalOffsetMatrix, aStFinalOutMatrix, aBalanceWheel
    };

	std::vector<MObject> driverObjs = {
        aStDriver, aStEditMode, aStRadius, aStUiData,
        aStName, aStLinkNameToNode
    };
	addAttributes<StrataPoint>(driverObjs);
    addAttributes<StrataPoint>(drivenObjs);

    setAttributesAffect<StrataPoint>(driverObjs, drivenObjs);
    

	CHECK_MSTATUS_AND_RETURN_IT(s);
	return s;
}

MStatus StrataPoint::computeDriver(MDataHandle& parentDH, MDataBlock& data) {
    // compute matrices for a single driver entry
    MStatus s(MS::kSuccess);
    
    // dispatch by driver type
    MMatrix driverMat;
    int driverType = parentDH.child(aStDriverType).asInt();
    switch (driverType){
        case STDriverType::point : 
            driverMat = parentDH.child(aStDriverPointMatrix).asMatrix();
            break;
        case STDriverType::line :
            DEBUGS("LINE NOT SUPPORTED YET");
            break;
        case STDriverType::face:
            DEBUGS("FACE NOT SUPPORTED YET");
            break;
    }
    parentDH.child(aStDriverOutMatrix).setMMatrix(driverMat);

    return MS::kSuccess;
}

MStatus combineDriverInfluences(StrataPoint& node,
    MDataBlock& data) {
    // assuming that all drivers have been computed in the data block,
    // combine all the offset matrices using their weights
    MStatus s(MS::kSuccess);

    MArrayDataHandle driversHdl = data.inputArrayValue(StrataPoint::aStDriver);
    float weightTotal = 0.0;
    for (uint i = 0; i < driversHdl.elementCount(); i++) {
        jumpToElement(driversHdl, i);
        //weightTotal += toelement  driversHdl
    }

    
    


    return s;
}

MStatus StrataPoint::compute(const MPlug& plug, MDataBlock& data) {
    MStatus s = MS::kSuccess;
    


    //if ((plug.attribute() == MPxTransform::matrix)
    //    || (plug.attribute() == MPxTransform::inverseMatrix)
    //    || (plug.attribute() == MPxTransform::worldMatrix)
    //    || (plug.attribute() == MPxTransform::worldInverseMatrix)
    //    || (plug.attribute() == MPxTransform::parentMatrix)
    //    || (plug.attribute() == MPxTransform::parentInverseMatrix))
    //{
    //    StrataPointMatrix xform(MMatrix::identity);
    //    computeLocalTransformation(&xform, data);
    //}

    //DEBUGS("point compute");

    //s = MPxTransform::compute(plug, data);
    //MCHECK(s, "stratapoint parent compute failed");


    //MHWRender::MRenderer::setGeometryDrawDirty(thisMObject(), false);

    if (data.isClean(plug)) {
        return s;
    }

    data.setClean(plug);

    return s;
}

void StrataPoint::postConstructor() {
    // check that this node's parent transform is named properly,
    // connect up its offsetParentMatrix attribute

    // this method fires before this shape node gets a parent,
    // so add a dag callback to check when it does get a parent added

    // TOO COMPLICATED just make the creation function in python
    
    // remove the default locator attributes from channelbox
    MFnAttribute aFn;
    MFnDagNode thisFn(thisMObject());

    //MObject scaleObj = thisFn.attribute("localScale");
    //if (scaleObj.isNull()) {
    //    DEBUGS("COULD NOT GET SCALE OBJ")
    //}
    //aFn.setObject(scaleObj);
    //DEBUGS(aFn.name());
    //thisFn.removeAttribute(scaleObj); ////// CRASHES :(

    /*
    for (auto i : "XYZ") {
        aFn.setObject(thisFn.attribute("localPosition" + i));
        aFn.setChannelBox(false);
        aFn.setKeyable(false);
        aFn.setObject(thisFn.attribute("localScale" + i));
        aFn.setChannelBox(false);
        aFn.setKeyable(false);
    }
    aFn.setObject(thisFn.attribute("localPosition"));
    aFn.setChannelBox(false);
    aFn.setKeyable(false);
    aFn.setObject(thisFn.attribute("localScale"));
    aFn.setChannelBox(false);
    aFn.setKeyable(false);

    for (auto i : "xyz") {
        aFn.setObject(thisFn.attribute("lp" + i));
        aFn.setChannelBox(false);
        aFn.setKeyable(false);
        aFn.setObject(thisFn.attribute("ls" + i));
        aFn.setChannelBox(false);
        aFn.setKeyable(false);
    }
    aFn.setObject(thisFn.attribute("lp"));
    aFn.setChannelBox(false);
    aFn.setKeyable(false);
    aFn.setObject(thisFn.attribute("ls"));
    aFn.setChannelBox(false);
    aFn.setKeyable(false);*/



    return;

    MStatus s(MS::kSuccess);
    MFnDagNode parentFn;
    
    MDGModifier dgMod;

    //DEBUGS("Point postConstructor() - has parent:");
    //DEBUGS(thisFn.parentCount());
    //DEBUGS(thisFn.dagPath().fullPathName());
    //if (thisFn.parentCount() == 0) {
    //    DEBUGS("NO PARENT");
    //    //MCallbackId creationCbId = MDagMessage::addParentAddedCallback()
    //    return;
    //}

    parentFn.setObject(thisFn.parent(0));

    MPlug matOutPlug = thisFn.findPlug(aStFinalOutMatrix, false, &s);
    CHECK_MSTATUS(s); // , "could not get final out plug to connect point shape");
    MPlug matInPlug = parentFn.findPlug("offsetParentMatrix", false, &s);
    CHECK_MSTATUS(s);

    dgMod.connect(matOutPlug, matInPlug);
    dgMod.doIt();


}

//MMatrix StrataPoint::asMatrix()
//MMatrix StrataPoint::getMatrix(MStatus* s) {
//    DEBUGS("call getMatrix");
//    return MPxTransform::getMatrix(s);
//}

//MStatus StrataPoint::computeLocalTransformation(MPxTransformationMatrix* xform, MDataBlock& data) {
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
//    xform->copyValues(&MPxTransformationMatrix(endMat));
//    DEBUGS("computed local transformation")
//    return s;
//
//}

MHWRender::DrawAPI StrataPointDrawOverride::supportedDrawAPIs() const
{
	// this plugin supports both GL and DX
	return (MHWRender::kOpenGL | MHWRender::kDirectX11 | MHWRender::kOpenGLCoreProfile);
}
bool StrataPointDrawOverride::isBounded(const MDagPath& /*objPath*/,
	const MDagPath& /*cameraPath*/) const
{
	return true;
}

float StrataPointDrawOverride::getMultiplier(const MDagPath& objPath) const
{
	// Retrieve value of the size attribute from the node
	// don't bother converting to distance and back, units are units
	MStatus status;
	MObject footprintNode = objPath.node(&status);
	if (status)
	{
		MPlug plug(footprintNode, StrataPoint::aStRadius);
		if (!plug.isNull())
		{
			return plug.asFloat();
		}
	}
	return 1.0f;
}

MBoundingBox StrataPointDrawOverride::boundingBox(
	const MDagPath& objPath,
	const MDagPath& cameraPath) const
{
	MPoint corner(-1, -1, -1);
	float multiplier = getMultiplier(objPath);
	return MBoundingBox(corner * multiplier, corner * multiplier * -1.0);
}


// Called by Maya each time the object needs to be drawn.
MUserData* StrataPointDrawOverride::prepareForDraw(
    const MDagPath& objPath,
    const MDagPath& cameraPath,
    const MHWRender::MFrameContext& frameContext,
    MUserData* oldData)
{
    // Any data needed from the Maya dependency graph must be retrieved and cached in this stage.
    // There is one cache data for each drawable instance, if it is not desirable to allow Maya to handle data
    // caching, simply return null in this method and ignore user data parameter in draw callback method.
    // e.g. in this sample, we compute and cache the data for usage later when we create the 
    // MUIDrawManager to draw footprint in method addUIDrawables().
    //DEBUGS("prepare for draw")
    StrataPointUserData* data = dynamic_cast<StrataPointUserData*>(oldData);
    if (!data)
    {
        data = new StrataPointUserData();
    }
    MStatus s = MS::kSuccess;

    float fMultiplier = getMultiplier(objPath);

    // get correct color and depth priority based on the state of object, e.g. active or dormant
    MObject node = objPath.node();
    data->fColor = MHWRender::MGeometryUtilities::wireframeColor(objPath);
    MFnDagNode thisFn(objPath);
    if (thisFn.parentCount() == 0) {
        return data;
    }
    MDagPath tfPath(objPath);
    tfPath.pop(1);
    MFnTransform transformFn(tfPath);
    MVector worldPos = transformFn.getTranslation(MSpace::kWorld, &s);
    CHECK_MSTATUS(s);
    data->pos = worldPos;

    if (!node.isNull()) {
        
        MPlug rgbPlug = MFnDependencyNode(node).findPlug("objectColorRGB", s);
        CHECK_MSTATUS(s);
        if (!rgbPlug.isNull()){
            //DEBUGS("getting rgb colour");
            // get RGB colour
            MColor rgb = MColor(
                rgbPlug.child(0).asFloat(),
                rgbPlug.child(1).asFloat(),
                rgbPlug.child(2).asFloat()
                );
            //DEBUGS(rgbPlug.child(0).name());
            data->fColor = rgb;
        }
        // fall back to positions if colour not set
        if (data->fColor == MColor::kOpaqueBlack) {
            worldPos = (worldPos.normal() + MVector(1.0, 1.0, 1.0)) / 2.0;
            data->fColor.set( MColor::kRGB, 
                float(worldPos[0]), float(worldPos[1]), float(worldPos[2]));
        }
        
        MPlug radiusPlug = MFnDependencyNode(node).findPlug(StrataPoint::aStRadius, s);
        CHECK_MSTATUS(s);
        data->size = radiusPlug.asFloat();
    }

    switch (MHWRender::MGeometryUtilities::displayStatus(objPath))
    {
    case MHWRender::kLead:
        data->fColor = data->fColor * 2.0;
        data->fDepthPriority = MHWRender::MRenderItem::sActiveWireDepthPriority;
        break;
    case MHWRender::kActive:
        data->fColor = data->fColor * 1.5;
        data->fDepthPriority = MHWRender::MRenderItem::sActiveWireDepthPriority;
        break;
    case MHWRender::kHilite:
        data->fColor = data->fColor * 1.5;
        data->fDepthPriority = MHWRender::MRenderItem::sActiveWireDepthPriority;
        break;
    //case MHWRender::kActiveComponent:
        
    default:
        data->fDepthPriority = MHWRender::MRenderItem::sDormantFilledDepthPriority;
        break;
    }
    return data;
}
// addUIDrawables() provides access to the MUIDrawManager, which can be used
// to queue up operations for drawing simple UI elements such as lines, circles and
// text. To enable addUIDrawables(), override hasUIDrawables() and make it return true.
void StrataPointDrawOverride::addUIDrawables(
    const MDagPath& objPath,
    MHWRender::MUIDrawManager& drawManager,
    const MHWRender::MFrameContext& frameContext,
    const MUserData* data)
{
    // Get data cached by prepareForDraw() for each drawable instance, then MUIDrawManager 
    // can draw simple UI by these data.
    //DEBUGS("addUIDrawables");
    StrataPointUserData* pLocatorData = (StrataPointUserData*)data;
    if (!pLocatorData)
    {
        DEBUGS("no prev user data")
        return;
    }
    // just draw a circle
    drawManager.beginDrawable();
    drawManager.setColor(pLocatorData->fColor);
    drawManager.setDepthPriority(pLocatorData->fDepthPriority);
    drawManager.setLineWidth(1.0);
    MPoint pos(0.0, 0.0, 0.0); // Position of the sphere
    if (frameContext.getDisplayStyle() & MHWRender::MFrameContext::kGouraudShaded) {
        drawManager.sphere(pos, pLocatorData->size, true);
    }
    else {
        MDoubleArray dir = frameContext.getTuple(MFrameContext::kViewDirection);
        MVector normal(dir[0], dir[1], dir[2]);
        drawManager.circle(pos, normal, pLocatorData->size, false);
    }
    drawManager.endDrawable();
}



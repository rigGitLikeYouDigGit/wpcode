
#pragma once
#include <vector>
//#include <array>
#include <maya/MMatrix.h>
#include <maya/MFnTransform.h>
#include <maya/MDrawRegistry.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MPointArray.h>
#include <maya/MGlobal.h>
#include <maya/MEventMessage.h>
#include <maya/MFnDependencyNode.h>

#include "macro.h"
#include "api.h"
#include "stratapoint.h"


using namespace ed;

MTypeId StrataPointMatrix::id(0x00122C1D);


MTypeId StrataPoint::kNODE_ID(0x00122C1C);
MString StrataPoint::kNODE_NAME("strataPoint");

MString StrataPoint::drawDbClassification("drawdb/geometry/stratapoint");
MString StrataPoint::drawRegistrantId("StrataPointPlugin");

MObject StrataPoint::aEditMode;
MObject StrataPoint::aRadius;

MStatus StrataPoint::initialize() {
	MStatus s = MS::kSuccess;
	MFnNumericAttribute nFn;
	aEditMode = nFn.create("editMode", "editMode", MFnNumericData::kBoolean, 0);
	nFn.setChannelBox(1);

	// radius only affects visuals for now
	aRadius = nFn.create("radius", "radius", MFnNumericData::kFloat, 0.1);
	nFn.setAffectsAppearance(true);
    nFn.setAffectsWorldSpace(true);
    nFn.setMin(0.0);
	nFn.setChannelBox(1);


	std::vector<MObject> driverObjs = { aEditMode };

	addAttributes<StrataPoint>(driverObjs);
	addAttribute(aRadius);
    attributeAffects(aRadius, aEditMode);
    attributeAffects(aEditMode, aRadius);

	CHECK_MSTATUS_AND_RETURN_IT(s);
	return s;
}

MStatus StrataPoint::compute(const MPlug& plug, MDataBlock& data) {
    MStatus s = MS::kSuccess;
    DEBUGS("point compute");
    MHWRender::MRenderer::setGeometryDrawDirty(thisMObject(), false);
    data.setClean(plug);

    //if (data.isClean(plug)) {
    //    return s;
    //}

    return s;
}

// drawing
//StrataPointDrawOverride::StrataPointDrawOverride(const MObject& obj)
//    : MHWRender::MPxDrawOverride(obj, NULL, false)
//{
//    /*fModelEditorChangedCbId = MEventMessage::addEventCallback(
//        "modelEditorChanged", OnModelEditorChanged, this);
//    MStatus status;
//    MFnDependencyNode node(obj, &status);
//    fFootPrint = status ? dynamic_cast<footPrint*>(node.userNode()) : NULL;*/
//}
//StrataPointDrawOverride::~StrataPointDrawOverride()
//{
//    /*fFootPrint = NULL;
//    if (fModelEditorChangedCbId != 0)
//    {
//        MMessage::removeCallback(fModelEditorChangedCbId);
//        fModelEditorChangedCbId = 0;
//    }*/
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
		MPlug plug(footprintNode, StrataPoint::aRadius);
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

    float fMultiplier = getMultiplier(objPath);

    // get correct color and depth priority based on the state of object, e.g. active or dormant
    MObject node = objPath.node();
    data->fColor = MHWRender::MGeometryUtilities::wireframeColor(objPath);
    MVector worldPos = MFnTransform(objPath).translation(MSpace::kWorld);
    data->pos = worldPos;

    if (!node.isNull()) {
        MS s = MS::kSuccess;
        MPlug rgbPlug = MFnDependencyNode(node).findPlug("objectColorRGB", s);
        CHECK_MSTATUS(s);
        if (!rgbPlug.isNull()){
            DEBUGS("getting rgb colour");
            // get RGB colour
            MColor rgb = MColor(
                rgbPlug.child(0).asFloat(),
                rgbPlug.child(1).asFloat(),
                rgbPlug.child(2).asFloat()
                );
            DEBUGS(rgbPlug.child(0).name());
            data->fColor = rgb;
        }
        // fall back to positions if colour not set
        if (data->fColor == MColor::kOpaqueBlack) {
            worldPos = (worldPos.normal() + MVector(1.0, 1.0, 1.0)) / 2.0;
            data->fColor.set( MColor::kRGB, worldPos[0], worldPos[1], worldPos[2]);
        }
        
        MPlug radiusPlug = MFnDependencyNode(node).findPlug(StrataPoint::aRadius, s);
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
        ////drawManager.mesh(MHWRender::MUIDrawManager::kTriangles, pLocatorData->fTriangleList);
        //filled = true;
    }
    else {
        DEBUGS("drawing circle");
        //drawManager.circle2d(pos, pLocatorData->size, false);
        //MVector normal = pLocatorData->pos - frameContext.getCurrentCameraPath().inclusiveMatrix().get()
        MDoubleArray dir = frameContext.getTuple(MFrameContext::kViewDirection);
        MVector normal(dir[0], dir[1], dir[2]);
        drawManager.circle(pos, normal, pLocatorData->size, false);
        
    }

    drawManager.endDrawable();
}



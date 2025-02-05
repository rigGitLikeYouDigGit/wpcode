
#pragma once
//#include <maya/MPxNode.h>
#include <maya/MPxTransform.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MEventMessage.h>
#include <maya/MGlobal.h>

class StrataPoint : public MPxTransform {
public:
	StrataPoint() {}
	virtual ~StrataPoint() {}

	static void* creator() {
		StrataPoint* newObj = new StrataPoint;
		return newObj;
	}
	//virtual void postConstructor();
	//virtual MStatus connectionMade(
	//	const MPlug& plug, const MPlug& otherPlug, bool asSrc);
	//virtual MStatus connectionBroken(
	//	const MPlug& plug, const MPlug& otherPlug, bool asSrc);

	/*static MTypeId kNODE_ID = MTypeId(0x00122C1C);
	static MString kNODE_NAME = MString("curveFrame");*/
	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;

	// attribute MObjects
	static MObject aEditMode;
	static MObject aRadius;

	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);


};

class StrataPointMatrix : public MPxTransformationMatrix
{

public:
	static  MTypeId id;
protected:
	typedef MPxTransformationMatrix ParentClass;
};

//draw override stuff copied directly from the footprint example
class StrataPointUserData : public MUserData
{
public:
	MColor       fColor{ 1.f, 0.f, 0.f, 1.f };
	unsigned int fDepthPriority;
	float size;
	MVector pos{ 0, 0, 0 };
};
class StrataPointDrawOverride : public MHWRender::MPxDrawOverride
{
public:

	// callback is used to detect changes in scene, without setting 
	// "isAlwaysDirty"
	// a bit complicated but there doesn't seem to be nicer way

	static void onModelEditorChanged(void* clientData) {
		DEBUGS("onModelEditorChanged");
		StrataPointDrawOverride* ovr = static_cast<StrataPointDrawOverride*>(clientData);
		if (!ovr) {
			DEBUGS("failed to cast");
		}
		if (!(ovr->pointNodePtr)) {
			DEBUGS("missing node pointer");
		}
		if (ovr && ovr->pointNodePtr)
		{
			MHWRender::MRenderer::setGeometryDrawDirty(ovr->pointNodePtr->thisMObject());
		}
	}
	MCallbackId fModelEditorChangedCbId = 0;

	// need a pointer to the transform, so we can get an MObject for setGeometryDirty
	// feels strange to hold a pointer in this override class, but apparently it's fine
	StrataPoint* pointNodePtr;

	static MHWRender::MPxDrawOverride* creator(const MObject& obj) {
		StrataPointDrawOverride* ptr = new StrataPointDrawOverride(
			obj);
		return ptr;
	}
	// setting alwaysDirty to true here, because I couldn't find another easy way to have it 
	// redraw when attributes changed, setting geometry dirty seemed to do nothing
	StrataPointDrawOverride(const MObject& obj) : MHWRender::MPxDrawOverride(
		obj, NULL, true) {
		DEBUGS("draw override init");
		fModelEditorChangedCbId = MEventMessage::addEventCallback(
			"modelEditorChanged", onModelEditorChanged, this);
		MStatus status;
		MFnDependencyNode node(obj, &status);
		pointNodePtr = status ? dynamic_cast<StrataPoint*>(node.userNode()) : NULL;

	}

	~StrataPointDrawOverride() {
		pointNodePtr = NULL;
		if (fModelEditorChangedCbId != 0)
		{
			MMessage::removeCallback(fModelEditorChangedCbId);
			fModelEditorChangedCbId = 0;
		}
	};
	MHWRender::DrawAPI supportedDrawAPIs() const override;
	bool isBounded(
		const MDagPath& objPath,
		const MDagPath& cameraPath) const override;
	MBoundingBox boundingBox(
		const MDagPath& objPath,
		const MDagPath& cameraPath) const override;
	MUserData* prepareForDraw(
		const MDagPath& objPath,
		const MDagPath& cameraPath,
		const MHWRender::MFrameContext& frameContext,
		MUserData* oldData) override;
	bool hasUIDrawables() const override { return true; }
	void addUIDrawables(
		const MDagPath& objPath,
		MHWRender::MUIDrawManager& drawManager,
		const MHWRender::MFrameContext& frameContext,
		const MUserData* data) override;

	

	float getMultiplier(const MDagPath& objPath) const;
//
//	MHWRender::MPxDrawOverride(obj, NULL, false)
//private:
//	StrataPointDrawOverride(const MObject& obj);
	/*float getMultiplier(const MDagPath& objPath) const;
	static void OnModelEditorChanged(void* clientData);
	footPrint* fFootPrint;
	MCallbackId fModelEditorChangedCbId;*/
};


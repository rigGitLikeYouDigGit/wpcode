
#pragma once
//#include <maya/MPxNode.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MEventMessage.h>
#include <maya/MGlobal.h>

class StrataPointNode : public MPxLocatorNode {
public:
	StrataPointNode() {}
	virtual ~StrataPointNode() {}

	static void* creator() {
		StrataPointNode* newObj = new StrataPointNode();
		return newObj;
	}
	virtual void postConstructor();
	//virtual MStatus connectionMade(
	//	const MPlug& plug, const MPlug& otherPlug, bool asSrc);
	//virtual MStatus connectionBroken(
	//	const MPlug& plug, const MPlug& otherPlug, bool asSrc);


	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);
	MStatus computeDriver(MDataHandle& parentDH, MDataBlock& data);
	
	// creation mechanism to fire once node is added to the dag
	static MCallbackId creationCbId;




	// attribute MObjects
	static MObject aStEditMode;
	static MObject aStRadius;
	static MObject aStName;
	static MObject aStLinkNameToNode;
	static MObject aBalanceWheel;
	
	static MObject aStDriver;
	static MObject aStDriverType;
	static MObject aStDriverWeight; // normalised, default of 1.0

	// forgive me for these essays of attribute names
	static MObject aStDriverClosestPoint;
	static MObject aStDriverUseClosestPoint;
	
	static MObject aStDriverPointMatrix;

	static MObject aStDriverCurve;
	static MObject aStDriverRefLengthCurve;
	// we assume whatever parametre is used on driver curve is used on up, too complicated if not
	static MObject aStDriverUpCurve; 
	static MObject aStDriverCurveLength;
	static MObject aStDriverCurveParam;
	static MObject aStDriverCurveLengthParamBlend;
	static MObject aStDriverCurveReverseBlend;
	static MObject aStDriverCurveNormLengthBlend;
	
	// use a generic attribute here,
	// accept bool from other strata surface, nurbs surface or poly
	static MObject aStDriverSurface;
	
	// normalized weight of this driver
	static MObject aStDriverNormalizedWeight;
	// final computed parent matrix for this driver
	static MObject aStDriverOutMatrix;
	// final offset for this point, from whatever active source is chosen
	static MObject aStDriverLocalOffsetMatrix;
	// should edit mode change the params of this driver in edit mode, or just the local offset
	static MObject aStDriverUpdateParamsInEditMode;
	
	// final matrix of all weighted drivers
	static MObject aStFinalDriverOutMatrix;
	// final local offset matrix, from final driver matrix
	static MObject aStFinalLocalOffsetMatrix;
	// final final matrix plug to feed into offsetParentMatrix of parent transform
	static MObject aStFinalOutMatrix;

	static MObject aStUiData;
	


};


//draw override stuff copied directly from the footprint example
class StrataPointNodeUserData : public MUserData
{
public:
	MColor       fColor{ 1.f, 0.f, 0.f, 1.f };
	unsigned int fDepthPriority = 1;
	float size = 1.0f;
	MVector pos{ 0, 0, 0 };
};
class StrataPointNodeDrawOverride : public MHWRender::MPxDrawOverride
{
public:

	// need a pointer to the transform, so we can get an MObject for setGeometryDirty
	// feels strange to hold a pointer in this override class, but apparently it's fine
	StrataPointNode* pointNodePtr = nullptr;

	static MHWRender::MPxDrawOverride* creator(const MObject& obj) {
		StrataPointNodeDrawOverride* ptr = new StrataPointNodeDrawOverride(
			obj);
		return ptr;
	}
	// setting alwaysDirty to true here, because I couldn't find another easy way to have it 
	// redraw when attributes changed, setting geometry dirty seemed to do nothing
	StrataPointNodeDrawOverride(const MObject& obj) : MHWRender::MPxDrawOverride(
		obj, NULL, true) {

	}

	~StrataPointNodeDrawOverride() {
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


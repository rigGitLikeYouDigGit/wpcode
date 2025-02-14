
#pragma once
//#include <maya/MPxNode.h>
#include <maya/MPxTransform.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MEventMessage.h>
#include <maya/MGlobal.h>

class StrataSurface : public MPxTransform {
	/* Strata surface, combining curves into lofted spans
	* BUT HOW
	* this should only govern the topology - NOT apply final offsets
	* and displancements to match a given polygon mesh
	* that will be a separate node.
	* 
	* generate a smooth polygon mesh
	* pack BLIND DATA on it per-point, noting the EARLIEST ELEMENT
	* that represents it in previous strata systems
	* this will be used in back-propagation to match a target mesh,
	* by actually moving the points/edges around.
	* 
	* for now I think this should also handle edge / mesh resolutions -
	* more correct to put it in the Sculpt node, but then to get a neutral poly
	* mesh, you would need
	* 
	* StrataSurface -> StrataSculpt (neutral) -> random maya deformers -> StrataSculpt (active)
	* 
	* BACK PROPAGATION - at each step, compare the END with the strata elements at the START -
	* we can easily insert deformers and nodes in-between, the strata elements are the only things
	* that propagation affects
	* 
	* 
	* node structure - 
	* this node is a transform, automatically creating a polygon mesh below it - this mesh will be
	* the output mesh.
	* by default, also listen on this mesh for any modelling activity - if edit mode is active,
	* backpropagate those changes, but we can't save any deltas on this node yet.
	* 
	* 
	*/


public:
	StrataSurface() {}
	virtual ~StrataSurface() {}

	static void* creator() {
		StrataSurface* newObj = new StrataSurface;
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

	typedef MPxTransform ParentClass;

	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);
	//MStatus computeDriver(MDataHandle& parentDH, MDataBlock& data);

	//virtual MStatus computeLocalTransformation(MPxTransformationMatrix* xform, MDataBlock& data);

	void postConstructor();


	// attribute MObjects
	static MObject aBalanceWheel;
	static MObject aStOutMesh;


	static MObject aStEditMode;
	static MObject aStUiData;


};

class StrataSurfaceMatrix : public MPxTransformationMatrix
{

public:
	StrataSurfaceMatrix(const MMatrix&);

	static  MTypeId id;
protected:
	typedef MPxTransformationMatrix ParentClass;
};

////draw override stuff copied directly from the footprint example
//class StrataSurfaceUserData : public MUserData
//{
//public:
//	MColor       fColor{ 1.f, 0.f, 0.f, 1.f };
//	unsigned int fDepthPriority;
//	float size;
//	MVector pos{ 0, 0, 0 };
//};
// 
//class StrataSurfaceDrawOverride : public MHWRender::MPxDrawOverride
//{
//public:
//
//	// need a pointer to the transform, so we can get an MObject for setGeometryDirty
//	// feels strange to hold a pointer in this override class, but apparently it's fine
//	StrataSurface* pointNodePtr;
//
//	static MHWRender::MPxDrawOverride* creator(const MObject& obj) {
//		StrataSurfaceDrawOverride* ptr = new StrataSurfaceDrawOverride(
//			obj);
//		return ptr;
//	}
//	// setting alwaysDirty to true here, because I couldn't find another easy way to have it 
//	// redraw when attributes changed, setting geometry dirty seemed to do nothing
//	StrataSurfaceDrawOverride(const MObject& obj) : MHWRender::MPxDrawOverride(
//		obj, NULL, true) {
//
//	}
//
//	~StrataSurfaceDrawOverride() {
//	};
//	MHWRender::DrawAPI supportedDrawAPIs() const override;
//	bool isBounded(
//		const MDagPath& objPath,
//		const MDagPath& cameraPath) const override;
//	MBoundingBox boundingBox(
//		const MDagPath& objPath,
//		const MDagPath& cameraPath) const override;
//	MUserData* prepareForDraw(
//		const MDagPath& objPath,
//		const MDagPath& cameraPath,
//		const MHWRender::MFrameContext& frameContext,
//		MUserData* oldData) override;
//	bool hasUIDrawables() const override { return true; }
//	void addUIDrawables(
//		const MDagPath& objPath,
//		MHWRender::MUIDrawManager& drawManager,
//		const MHWRender::MFrameContext& frameContext,
//		const MUserData* data) override;
//
//
//
//	float getMultiplier(const MDagPath& objPath) const;
//	//
//	//	MHWRender::MPxDrawOverride(obj, NULL, false)
//	//private:
//	//	StrataSurfaceDrawOverride(const MObject& obj);
//		/*float getMultiplier(const MDagPath& objPath) const;
//		static void OnModelEditorChanged(void* clientData);
//		footPrint* fFootPrint;
//		MCallbackId fModelEditorChangedCbId;*/
//};
//

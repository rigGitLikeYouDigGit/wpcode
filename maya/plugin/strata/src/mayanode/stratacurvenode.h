
#pragma once
//#include <maya/MPxNode.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MPxSurfaceShape.h>
#include <maya/MEventMessage.h>
#include <maya/MGlobal.h>

//class StrataCurve : public MPxTransform {
//class StrataCurve : public MPxSurfaceShape {
class StrataCurve : public MPxNode {
	/* curve component of Strata, connecting points to form faces.
	* In Maya, this is a transform node that runs the various Strata functions and outputs
	* a resulting curve to its own shape - this can be passed through other maya rig
	* functions before being fed into a Strata mesh, to inject control if needed.
	* transform again for clarity and ease in the outliner, shouldn't need to touch
	* the curve shape frequently, outside of manual edits
	* 
	* IN FUTURE maybe rewrite this with a plugin shape, but I don't think it needs it
	* 
	* going with basic MPxNode here, which makes it more annoying to select in channelbox,
	* but for now it's fine, we can solve that in UI.
	* MPxSurfaceShape is a LOT of work, and the StrataSurface will need it anyway - no need to
	* copy %90 of NurbsCurve here too
	*/


public:
	StrataCurve() {}
	virtual ~StrataCurve() {}

	static void* creator() {
		StrataCurve* newObj = new StrataCurve;
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

	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);
	//MStatus computeDriver(MDataHandle& parentDH, MDataBlock& data);

	//virtual MStatus computeLocalTransformation(MPxTransformationMatrix* xform, MDataBlock& data);

	void postConstructor();


	// attribute MObjects

	// input point attributes

	static MObject aStStartMatrix;
	static MObject aStStartTangent;
	static MObject aStStartUp;
		static MObject aStStartUpX;
		static MObject aStStartUpY;
		static MObject aStStartUpZ;
	static MObject aStStartTangentScale;
	static MObject aStStartAutoTangentBlend;

	static MObject aStEndMatrix;
	static MObject aStEndTangent;
	static MObject aStEndUp;
	static MObject aStEndTangentScale;
	static MObject aStEndAutoTangentBlend;

	static MObject aStNSpans;
	static MObject aStDegree;

	static MObject aStEditMode;
	static MObject aStUiData;

	static MObject aBalanceWheel; // used to recover node object outside of graph
	static MObject aStOutCurve;
	static MObject aStOutUpCurve;


};




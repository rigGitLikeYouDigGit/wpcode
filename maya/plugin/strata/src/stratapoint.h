
#pragma once
//#include <maya/MPxNode.h>
#include <maya/MPxTransform.h>

#include <maya/MFnNumericAttribute.h>


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

	// attribute MObjects
	static MObject aEditMode;

	static MStatus initialize();

	virtual MStatus compute(
		const MPlug& plug, MDataBlock& data) {
		MStatus s = MS::kSuccess;
		if (data.isClean(plug)) {
			return s;
		}
		data.setClean(plug);
		return s;
	}

};

//MTypeId StrataPoint::kNODE_ID(0x00122C1C);
//MString StrataPoint::kNODE_NAME("curveFrame");
//MTypeId StrataPoint::kNODE_ID = MTypeId(0x00122C1C);
//MString StrataPoint::kNODE_NAME("curveFrame");





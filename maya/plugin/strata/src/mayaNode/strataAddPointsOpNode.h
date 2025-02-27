#pragma once


#include "../MInclude.h"
#include "strataOpNodeBase.h"

#include "strataMayaLib.h"


class StrataAddPointsOpNode : public MPxNode, public StrataOpMixin {
public:
	StrataAddPointsOpNode() {}
	virtual ~StrataAddPointsOpNode() {}

	static void* creator() {
		StrataAddPointsOpNode* newObj = new StrataAddPointsOpNode();
		return newObj;
	}
	virtual void postConstructor();


	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);
	MStatus computeDriver(MDataHandle& parentDH, MDataBlock& data);


	DECLARE_STRATA_STATIC_MEMBERS


	static MStatus legalConnection(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	);


	MStatus connectionMade(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	MStatus connectionBroken(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);


};


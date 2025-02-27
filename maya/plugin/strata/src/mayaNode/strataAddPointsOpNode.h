#pragma once


#include "../MInclude.h"
#include "strataOpNodeBase.h"

#include "strataMayaLib.h"
#include "../strataop/strataAddPointsOp.h"


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

	static MObject aStPoint; // compound attr for incoming points
	// do we just make this a bool attribute and do balancewheel tracking - 
	// keep it simple for now, connect each fully by attribute?
	// but we will still need to crawl DG for back-propagation and point fitting
	static MObject aStPointLocalMatrix; // locally transformed matrix of point
	static MObject aStPointHomeMatrix; // home or default matrix of point
	static MObject aStPointName; // matrix of point

	static MObject aStResult; // return the ordered global indices of newly created points
	// this is NOT the formal output of the node in Strata, that's the main geometry stream of the node itself
	// "result" is a node-local group collecting all elements (points) created in this operation

	typedef ed::StrataAddPointsOp strataOpType;

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


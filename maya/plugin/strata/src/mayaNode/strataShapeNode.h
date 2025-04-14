#pragma once


#include "../MInclude.h"
#include "strataMayaLib.h"
#include "../strataop/elementOp.h"
#include "strataOpNodeBase.h"

#include "../exp/expParse.h"

#include <maya/MPxSurfaceShapeUI.h>

/*
shape node to display end result of strata graph.
also allows interacting with points as if a normal maya shape,
creating data overrides,
or outputting spatial data back into maya

registering a new interactive shape class in maya, how hard can it be?

*/

# define NODE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStData;\
prefix MObject nodeT aStExp;\
prefix MObject nodeT aStMatrix;\
prefix MObject nodeT aStDriverMatrix;\
prefix MObject nodeT aStWorldspace;\


class StrataShapeUI : public MPxSurfaceShapeUI {
	/* this class is apparently only for viewport1, 
	and deprecated in newer versions, but we still
	need a creator function to register it
	*/
public:
	static void* creator() {
		return new StrataShapeUI;
	}
	StrataShapeUI() {}
	~StrataShapeUI() {}
	void getDrawRequests(const MDrawInfo& info,
		bool objectAndActiveOnly,
		MDrawRequestQueue& queue) override 
	{
		return;
	};
};

//class StrataShapeNode : public MPxNode, public StrataOpNodeTemplate<ed::StrataElementOp> {
//class StrataShapeNode : public MPxNode, public StrataOpNodeBase {
class StrataShapeNode : public MPxComponentShape, public StrataOpNodeBase {
public:
	//using thisStrataOpT = ed::StrataElementOp;
	//using superT = StrataOpNodeTemplate<ed::StrataElementOp>;
	using superT = StrataOpNodeBase;
	using thisT = StrataShapeNode;
	StrataShapeNode() {}
	virtual ~StrataShapeNode() {}

	static void* creator() {
		StrataShapeNode* newObj = new StrataShapeNode();
		return newObj;
	}

	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);
	DECLARE_STATIC_NODE_H_MEMBERS(NODE_STATIC_MEMBERS);

	//virtual void postConstructor();

	//static MStatus legalConnection(
	//	const MPlug& plug,
	//	const MPlug& otherPlug,
	//	bool 	asSrc,
	//	bool& isLegal
	//);

	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	virtual MStatus syncStrataParams(MObject& nodeObj, MDataBlock& data);

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

	// override base class static strata objects, so each leaf class still has attributes
	// initialised separately to the base
	//DECLARE_STRATA_STATIC_MEMBERS;

	/*DECLARE_STATIC_NODE_MEMBERS(
		STRATAADDPOINTSOPNODE_STATIC_MEMBERS)*/

	void postConstructor();

	MStatus legalConnection(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	) const;

	virtual MStatus connectionMade(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	virtual MStatus connectionBroken(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

};


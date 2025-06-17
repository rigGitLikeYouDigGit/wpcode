#pragma once


#include "../MInclude.h"
#include "strataMayaLib.h"
#include "../strataop/mergeOp.h"
#include "strataOpNodeBase.h"

#include "../exp/expParse.h"

/*
shape node to display end result of strata graph.
also allows interacting with points as if a normal maya shape,
creating data overrides,
or outputting spatial data back into maya

registering a new interactive shape class in maya, how hard can it be?

*/

# define NODE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStDataIn;\
prefix MObject nodeT aStExpIn;\
prefix MObject nodeT aStSpaceModeIn;\
prefix MObject nodeT aStSpaceIndexIn;\
prefix MObject nodeT aStSpaceNameIn;\
prefix MObject nodeT aStMatrixIn;\
prefix MObject nodeT aStUVNIn;\
\
prefix MObject nodeT aStDataOut;\
prefix MObject nodeT aStExpOut;\
prefix MObject nodeT aStMatrixOut;\
prefix MObject nodeT aStCurveOut;\
\
prefix MObject nodeT aStShowPoints;\


/* per-space stuff - 
if space is -1, default, use either no space (if el has none)
or the first space of the element -
this should suffice for most of them.

if space name is given, use that if found; no effect if not
if space index is given, use that if found; no effect if not
*/

/* todo:
for visibility, enabled, allow setting expression based overrides
to show/hide groups and faces more precisely*/

namespace ed {



}



//class StrataShapeNode : public MPxNode, public StrataOpNodeTemplate<ed::StrataMergeOp> {
//class StrataShapeNode : public MPxNode, public StrataOpNodeBase {
//class StrataShapeNode : public MPxComponentShape, public StrataOpNodeTemplate<ed::StrataMergeOp> {
class StrataShapeNode : public MPxSurfaceShape, public StrataOpNodeTemplate<ed::StrataMergeOp> {
public:
	
	using superT = StrataOpNodeTemplate<ed::StrataMergeOp>;
	using thisT = StrataShapeNode;
	using thisStrataOpT = ed::StrataMergeOp;

	// cached values used for drawing
	float pointOpacity = 1.0;


	StrataShapeNode() {}
	virtual ~StrataShapeNode() {}

	static void* creator() {
		StrataShapeNode* newObj = new StrataShapeNode();
		return newObj;
	}

	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);
	DECLARE_STATIC_NODE_H_MEMBERS(NODE_STATIC_MEMBERS);


	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	MStatus addDeltaTarget(
		MObject& nodeObj, MDataBlock& data, MDataHandle& elDH,
		ed::StrataManifold& manifold, ed::SElement* finalEl, ed::SAtomBackDeltaGroup& deltaGrp
	);

	MStatus runShapeBackPropagation(MObject& nodeObj, MDataBlock& data);

	MStatus populateOutputs(MDataBlock& data);

	virtual MStatus syncStrataParams(MObject& nodeObj, MDataBlock& data);

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

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



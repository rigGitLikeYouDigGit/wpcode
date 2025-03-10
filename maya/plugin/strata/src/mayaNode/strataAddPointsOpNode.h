#pragma once


#include "../MInclude.h"
#include "strataMayaLib.h"
#include "strataOpNodeBase.h"
#include "../strataop/strataAddPointsOp.h"


/// experiment for making declaring / defining attr mobjects less painful
# define STRATAADDPOINTSOPNODE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStPoint;\
prefix MObject nodeT aStPointName;\
prefix MObject nodeT aStPointWorldMatrix;\
prefix MObject nodeT aStPointFinalDriverOutMatrix;\
prefix MObject nodeT aStPointFinalLocalOffsetMatrix;\


// create lines of the form 'static MObject aStPoint;'
# define DECLARE_STATIC_NODE_MEMBERS(attrsMacro) \
	attrsMacro(static, )

// create lines of the form 'MObject StrataAddPointsOpNode::aStPoint;'
# define DEFINE_STATIC_NODE_MEMBERS(attrsMacro, nodeT) \
	attrsMacro( , nodeT::)

	

class StrataAddPointsOpNode : public StrataOpNodeBase {
public:
	StrataAddPointsOpNode() {}
	virtual ~StrataAddPointsOpNode() {}

	static void* creator() {
		StrataAddPointsOpNode* newObj = new StrataAddPointsOpNode();
		return newObj;
	}
	//virtual void postConstructor();

	static MStatus legalConnection(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	);

	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

	// override base class static strata objects, so each leaf class still has attributes
	// initialised separately to the base
	DECLARE_STRATA_STATIC_MEMBERS;

	DECLARE_STATIC_NODE_MEMBERS(
		STRATAADDPOINTSOPNODE_STATIC_MEMBERS)


	typedef ed::StrataAddPointsOp strataOpType;



};


#pragma once

#include <maya/MObject.h>
#include "../MInclude.h"
#include "../stratacore/op.h"
#include "../stratacore/opgraph.h"


/*
Base class for all strata operation nodes - for now we try to mirror the maya graph 1:1 
in structure, if not always in evaluation

- one master plug for the graph flow

keep evaluation in step for now too, otherwise Undo gets insane.

don't literally inherit from maya base node class, just use mpxnode inheritAttributes

...can we have only one maya node class, and dynamically rearrange the attributes
for different Strata ops?
Maybe

stOutput is now an integer attribute for the output node index

*/

//#define DECLARE_VIRTUAL_STRATA_STATIC_MEMBERS(prefix)  \
//prefix static MObject aStGraph; \
//prefix static MObject aStParent;  \
//prefix static MObject aStInput; \
//prefix static MObject aStInputAlias; \
//prefix static MObject aStOpIndex; \
//prefix static MObject aStOutput; \
//prefix static MObject aStManifoldData; \
//prefix static MObject aStParam;\
//prefix static MObject aStParamExp; \
//prefix static MObject aStElData; \


# define STRATABASE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStInput; \
prefix MObject nodeT aStOpName; \
prefix MObject nodeT aStOutput; \


// create lines of the form 'static MObject aStPoint;'
# define DECLARE_STATIC_NODE_H_MEMBERS(attrsMacro) \
attrsMacro(static, )

// create lines of the form 'MObject StrataElementOpNode::aStPoint;'
# define DEFINE_STATIC_NODE_CPP_MEMBERS(attrsMacro, nodeT) \
attrsMacro( , nodeT::)




/// after all, why not
/// why shouldn't we inherit a base class from MPxNode
//struct StrataOpNodeBase : public MPxNode {
struct StrataOpNodeBase {
	/* mixin class to be inherited by all maya nodes that
	represent a single op in op graph

	can't have this be an MPxNode if we want to have common attributes across DG nodes and maya shapes
	*/

	/* each maya node creates a copy of an incoming strata graph, or
	* an empty graph.
	* Then the maya node creates one new strata op in the graph.
	* We have to use one pointer lookup from the incoming graph, but otherwise it's all DG-legal
	* 
	* stInput is an int array, and we only look for entry 0 as an input for the graph
	* 
	*/


	std::unique_ptr<ed::StrataOpGraph> opGraphPtr;
	int strataOpIndex = -1;

	using StrataOpT = ed::StrataOp;

	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);

	static const std::string getOpNameFromNode(MObject& nodeObj);

	StrataOpT* getStrataOp() {
		// return pointer to the current op, in this node's graph
		ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(strataOpIndex);
		if (nodePtr == nullptr) {
			return nullptr;
		}
		return static_cast<StrataOpT*>(nodePtr);
	}

	MStatus refreshGraphPtr(ed::StrataOpGraph* other);

	ed::StrataOp* createNewNode(MObject& mayaNodeMObject);
	
	static MStatus addStrataAttrs(
		std::vector<MObject>& driversVec,
		std::vector<MObject>& drivenVec
	);


	// probably a way to get access to the attr MObjects in this mixin's scope, 
	// but by contrast I actually understand this way with the template
	template <typename T>
	static void setOpIndexOnMayaNode(int opIndex, MObject& thisNode) {
		// update the maya attribute to this struct's op index
		MFnDependencyNode thisFn(thisNode);
		MPlug opIndexPlug = thisFn.findPlug(thisNode, T::aStOutput, false);
		opIndexPlug.setInt(opIndex);
	}

	Status syncOp(ed::StrataOp* op, MDataBlock& data) {
		/* update op from maya node datablock -
		this should be good enough, updating raw from MObject and plugs
		seems asking for trouble

		also set topoDirty / dataDirty flags here

		can't run this on newly created op directly, need to wait for compute
		*/
		return Status();
	}

	MStatus syncOpInputs(ed::StrataOp* op, const MObject& node);

	// shared compute function for all op nodes
	MStatus compute(const MPlug& plug, MDataBlock& data);

	void onInputConnectionChanged(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc);

	void postConstructor();

	static MStatus legalConnection(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	);

	virtual MStatus connectionMade(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	virtual MStatus connectionBroken(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);



};


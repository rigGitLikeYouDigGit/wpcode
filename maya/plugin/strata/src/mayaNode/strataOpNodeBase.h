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

#define DECLARE_STRATA_STATIC_MEMBERS \
static MObject aStGraph; \
static MObject aStParent;  \
static MObject aStInput; \
static MObject aStInputAlias; \
static MObject aStOpIndex; \
static MObject aStOutput; \
static MObject aStManifoldData; \
static MObject aStParam;\
static MObject aStParamExp; \
static MObject aStElData; \



#define DEFINE_STRATA_STATIC_MOBJECTS(NODETYPE) \
MObject NODETYPE::aStGraph; \
MObject NODETYPE::aStParent; \
\
MObject NODETYPE::aStInput; \
MObject NODETYPE::aStInputAlias; \
\
MObject NODETYPE::aStOpIndex; \
MObject NODETYPE::aStOutput; \
MObject NODETYPE::aStManifoldData; \
\
MObject NODETYPE::aStParam; \
MObject NODETYPE::aStParamExp; \
\
MObject NODETYPE::aStElData; \




/// after all, why not
/// why shouldn't we inherit a base class from MPxNode
struct StrataOpNodeBase : public MPxNode {
	/* mixin class to be inherited by all maya nodes that
	represent a single op in op graph

	was too awkward to make this an actual MPxNode base class

	INHERITING STATIC MOBJECTS -
	we declare static MObjects in the mixin here, then REDECLARE them
	in each concrete child -
	this hopefully lets us do StrataOpMixin::aStGraph on a pointer, and have it
	get the right MObject?
	we want the SAME NAMES, but DIFFERENT VALUES
	except no, because we don't know the child class, so StrataOpMixin::aStGraph will just
	return the class-level static object OF THIS MIXIN CLASS

	hmmmmmm
	*/

	/* if this node is connected to its graph, both of these will
	be populated - if not, both will be null.

	Maya node connection causes new op object to be instantiated and
	ownership passed to graph -
	pointers are populated with result

	testing shared_ptrs here JUST IN CASE maya does some weird time travel / object lifetime
	stuff in multithreading nodes, DG context etc -
	we still cull pointers on connect/disconnect, but this should catch the case where a node
	evaluates EXACTLY AS it's disconnected (somehow)

	weak pointers show better - this node doesn't OWN anything, only refers into
	the graph data store
	*/

	// ok I have a great idea
	// just don't delete the graph while the graph is running
	std::weak_ptr<ed::StrataOpGraph> opGraphPtr;

	typedef ed::StrataOp strataOpType; // redefine for explicit linking maya node type to strata Op

	strataOpType* opPtr = nullptr;

	DECLARE_STRATA_STATIC_MEMBERS;


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

	//StrataOp* createNewOp() {
	ed::StrataOp createNewOp() {
		// return a full op instance for this node - 
		// override in a dynamic node with the main if/branch logic, everything
		// after this should only deal with the base
		return ed::StrataOp();
		// pointer passed straight to make_unique, so should be safe?
		//return new StrataOp;
	}

	void syncOp(ed::StrataOp* op, MDataBlock& data) {
		/* update op from maya node datablock -
		this should be good enough, updating raw from MObject and plugs
		seems asking for trouble

		also set topoDirty / dataDirty flags here

		can't run this on newly created op directly, need to wait for compute
		*/
	}

	MStatus syncOpInputs(ed::StrataOp* op, const MObject& node);

	// shared compute function for all op nodes
	MStatus compute(const MPlug& plug, MDataBlock& data);

	void onInputConnectionChanged(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc);

	void postConstructor() {
		/* ensure graph pointer is reset*/
		opGraphPtr.reset();
	}

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


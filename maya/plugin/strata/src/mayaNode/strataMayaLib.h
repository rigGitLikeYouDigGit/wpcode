#pragma once

#include <memory>

#include "../MInclude.h"
#include "../macro.h"
#include "../api.h"

#include "../stratacore/manifold.h"
#include "../stratacore/op.h"
#include "../stratacore/opgraph.h"

/* used to ensure all strata op nodes have the same attribute objects
declared for connecting to the graph -
macros used to declare the same static MObjects in every class
*/


#define DECLARE_STRATA_STATIC_MEMBERS  \
static MObject aStGraph; \
static MObject aStParent;  \
static MObject aStInput; \
static MObject aStInputAlias; \
static MObject aStOpIndex; \
static MObject aStOutput; \
static MObject aStManifoldData; \
static MObject aStParam;\
static MObject aStParamExpression;


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
MObject NODETYPE::aStParamExpression; 


template<typename T>
MStatus addStrataAttrs(
	std::vector<MObject>& driversVec,
	std::vector<MObject>& drivenVec
) {
	MS s(MS::kSuccess);

	MFnNumericAttribute nFn;
	MFnCompoundAttribute cFn;
	MFnTypedAttribute tFn;

	// all nodes connected directly to master graph node
	T::aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kBoolean);
	nFn.setReadable(false);
	nFn.setStorable(false);
	nFn.setChannelBox(false);
	nFn.setKeyable(false);

	T::aStInput = nFn.create("stInput", "stInput", MFnNumericData::kInt);
	nFn.setReadable(false);
	nFn.setArray(true);
	nFn.setUsesArrayDataBuilder(true);
	nFn.setIndexMatters(true);
	nFn.setDefault(-1);
	// strata inputs use physical order always - there's no reason ever to have an empty entry here
	nFn.setDisconnectBehavior(MFnAttribute::kDelete);

	//T::aStInputAlias = tFn.create("stInputAlias", "stInputAlias", MFnData::kString);
	//tFn.setReadable(false);
	//tFn.setArray(true);
	//tFn.setUsesArrayDataBuilder(true);
	//tFn.setIndexMatters(true);

	// output index of st op node, use as output to all 
	// -1 as default so it's obvious when a node hasn't been initialised, connected to graph, etc
	T::aStOutput = nFn.create("stOutput", "stOutput", MFnNumericData::kInt, -1);
	nFn.setReadable(true);
	nFn.setWritable(false);

	//// add attributes

	std::vector<MObject> drivers = {
		T::aStGraph,
		T::aStInput
	};
	driversVec.insert(driversVec.end(), drivers.begin(), drivers.end());

	std::vector<MObject> driven = {
		T::aStOutput
	};
	drivenVec.insert(drivenVec.end(), driven.begin(), driven.end());

	return s;
}


/// after all, why not
/// why shouldn't we inherit a base class from MPxNode
struct StrataOpMixin : public MPxNode {
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

	/* CAN WE have a single node shell class, to dynamically create any op type?
	- we can have a pointer of the base StrataOp* type
	- don't have to downcast necessarily, just call overridden functions?
	*/


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
	ed::StrataOp createNewOp(){
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

	MStatus syncOpInputs(ed::StrataOp* op, const MObject& node) {
		// check through input plugs on maya node, 
		// triggered when input connections change
		op->inputs.clear();

		MFnDependencyNode depFn(node);
		MPlug inputArrPlug = depFn.findPlug("stInput", true);

		for (unsigned int i = 0; i < inputArrPlug.numConnectedElements(); i++) {
			int inId = inputArrPlug.connectionByPhysicalIndex(i).asInt();
			if (inId == -1) {
				continue;
			}
			op->inputs.push_back(inId);
		}
		

		// TODO: update input aliases too
		// later

		// flag graph as dirty
		opGraphPtr.lock()->nodeInputsChanged(op->index);

		return MS::kSuccess;
	}

	// shared compute function for all op nodes
	template <typename T>
	MStatus strataCompute(const MPlug& plug, MDataBlock& data) {
		MS s(MS::kSuccess);

		// check if plug is already computed
		if (data.isClean(plug)) {
			return s;
		}
		// check if graph connection has been lost
		if (opGraphPtr.expired()) {
			data.setClean(plug);
			return s;
		}

		// check if input data/indices have changed
		if (!data.isClean(T::aStInput)) {

		}

	}


	
	static bool checkIndexInputConnection(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc
	)		 {
		// check that an input to the array only comes from another strata maya node - 
		// can't just connect a random integer
		
		if (otherPlug.node() == inputArrayPlug.node()) { // no feedback loops
			return false;
		}
		if (MFnAttribute(otherPlug.attribute()).name() == "stOutput") {
			return true;
		}
		return false;
	}

	void onInputConnectionChanged(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc) {
		/*fire on connection and disconnection to input array - 
		update topology in graph, and mark graph as changed*/

	}

	void opNodePostConstructor() {
		/* ensure graph pointer is reset*/
		opGraphPtr.reset();
	}



};

#pragma once

#include <maya/MObject.h>
#include "../MInclude.h"
#include "../stratacore/op.h"
#include "../stratacore/opgraph.h"

#include "../strataop/elementOp.h" // TEMP


/*
Base class for all strata operation nodes - for now we try to mirror the maya graph 1:1 
in structure, if not always in evaluation

- one master plug for the graph flow

keep evaluation in step for now too, otherwise Undo gets insane.

don't literally inherit from maya base node class, just use mpxnode inheritAttributes

...can we have only one maya node class, and dynamically rearrange the attributes
for different Strata ops?
Maybe

we also have an explicit node connection for stGraph input, separate to the op inputs - 
in this way we enforce that ops are ADDED one by one, but we can still describe parallel graph structures.

stOutput is now an integer attribute for the output node index


TODO:
later, flag if graph input structure has changed, or only its data - 
if only data, only need to copy that data and re-eval the graph


Can you reuse a static MObject defined across multiple base classes?
feels like no
but also, how would Maya actually tell?

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
prefix MObject nodeT aStGraph; \
prefix MObject nodeT aStInput; \
prefix MObject nodeT aStOpName; \
prefix MObject nodeT aStOpNameOut; \
prefix MObject nodeT aStOutput; \


//// create lines of the form 'static MObject aStPoint;'
//# define DECLARE_STATIC_NODE_H_MEMBERS(attrsMacro) \
//attrsMacro(static, )
//
//// create lines of the form 'MObject StrataElementOpNode::aStPoint;'
//# define DEFINE_STATIC_NODE_CPP_MEMBERS(attrsMacro, nodeT) \
//attrsMacro( , nodeT::)




/// after all, why not
/// why shouldn't we inherit a base class from MPxNode
//struct StrataOpNodeBase : public MPxNode {

// here we template the base class, so each Maya node is specialised for a certain kind of Strata op?
// TEMPLATING BASE CLASS CAUSED GREAT PAIN
// now base class is real, an INTERMEDIATE class is templated, and that gets inherited into final Maya node class, alongside MPxNode etc

/* we pass a MObject& nodeObj param to all methods in the abstract bases,
to avoid any ambiguity with the proper MPxNode methods
*/
struct StrataOpNodeBase {

	using thisStrataOpT = ed::StrataOp;

	/* for some reason, createNode triggers connectionMade before even
	* postConstructor is called - 
	* this is here as a safeguard to do nothing til this node
	* is properly created in Maya
	* */
	bool addedToGraph = false; 

	// maya node now owns its proper strata graph
	std::shared_ptr<ed::StrataOpGraph> opGraphPtr;
	// pointer to an incoming graph, so we don't have to constantly walk the DG
	// this is also nulled by hand when connection is broken

	// weak pointer to the graph we need to copy 
	std::weak_ptr<ed::StrataOpGraph> sourceGraphPtr;


	// semantic/logical plug index to incoming graph
	std::map<int, std::weak_ptr<ed::StrataOpGraph>> incomingGraphPtrs;
	// weak anyway, just in case
	/* NB this means we'll have to merge multiple incoming graphs when we get to using
	multiple inputs - probably fine*/


	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);

	template <typename NodeT>
	static const std::string getOpNameFromNode(MObject& nodeObj);

	template <typename NodeT>
	static inline const int getOpIndexFromNode(const MObject& nodeObj);


	static const std::string getOpNameFromNode(MObject& nodeObj);

	static inline const int getOpIndexFromNode(const MObject& nodeObj);

	virtual MStatus setFreshGraph(MObject& nodeObj);
	MStatus syncIncomingGraphConnections(MObject& nodeObj);
	MStatus syncIncomingGraphData(MObject& nodeObj);

	void postConstructor(MObject& nodeObj);

	static int strataAttrsDefined;

	//template<typename NodeT>
	static MStatus defineStrataAttrs() {

		MS s(MS::kSuccess);

		MFnNumericAttribute nFn;
		MFnCompoundAttribute cFn;
		MFnTypedAttribute tFn;

		// all nodes connected directly to master graph node
		//NodeT::aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kInt);
		aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kInt);
		nFn.setReadable(false);
		nFn.setStorable(false);
		nFn.setChannelBox(false);
		nFn.setKeyable(false);

		//NodeT::aStInput = nFn.create("stInput", "stInput", MFnNumericData::kInt);
		aStInput = nFn.create("stInput", "stInput", MFnNumericData::kInt);
		nFn.setReadable(false);
		nFn.setArray(true);
		nFn.setUsesArrayDataBuilder(true);
		nFn.setIndexMatters(true);
		nFn.setDefault(-1);
		// strata inputs use physical order always - there's no reason ever to have an empty entry here
		// YES THERE IS NB - some nodes might only want aux inputs, but still create a fresh geo stream
		//nFn.setDisconnectBehavior(MFnAttribute::kDelete);
		nFn.setDisconnectBehavior(MFnAttribute::kReset);

		//NodeT::aStOpName = tFn.create("stOpName", "stOpName", MFnData::kString);
		aStOpName = tFn.create("stOpName", "stOpName", MFnData::kString);
		tFn.setDefault(MFnStringData().create(""));
		
		aStOpNameOut = tFn.create("stOpNameOut", "stOpNameOut", MFnData::kString);
		tFn.setDefault(MFnStringData().create(""));
		tFn.setWritable(false);
		tFn.setChannelBox(false);
		tFn.setStorable(false);



		//T::aStInputAlias = tFn.create("stInputAlias", "stInputAlias", MFnData::kString);
		//tFn.setReadable(false);
		//tFn.setArray(true);
		//tFn.setUsesArrayDataBuilder(true);
		//tFn.setIndexMatters(true);

		// output index of st op node, use as output to all 
		// -1 as default so it's obvious when a node hasn't been initialised, connected to graph, etc
		//NodeT::aStOutput = nFn.create("stOutput", "stOutput", MFnNumericData::kInt, -1);
		aStOutput = nFn.create("stOutput", "stOutput", MFnNumericData::kInt, -1);
		nFn.setReadable(true);
		nFn.setWritable(false);
		nFn.setChannelBox(false);
		nFn.setKeyable(false);

		// PARAMETRES
		/* we add the top one first, and assume each one will have a string expression?
		no idea whatsoever*/
		//aStParam = cFn.create("stParam", "stParam");
		//aStParamExp = tFn.create("stParamExp", "stParamExp", MFnData::kString);
		//tFn.setDefault(MFnStringData().create(""));
		//cFn.addChild(aStParamExp);

		//// ELEMENT DATA
		//// specific nodes naturally need to add their own inputs here
		//aStElData = cFn.create("stElData", "stElData");

		//// add attributes
		return s;
	}
	static MStatus addStrataAttrs(
		std::vector<MObject>& driversVec,
		std::vector<MObject>& drivenVec,
		std::vector<MObject>& toAddVec
	) { 
		/* I was physically unable to put this in the cpp file.
		* kept giving unresolved external symbols.
		* I developed a desire to unresolve myself
		*/
		MStatus s;
		if (!strataAttrsDefined) {
			s = defineStrataAttrs();
			MCHECK(s, "error DEFINING strata attrs");
			strataAttrsDefined = 1;
		}

		std::vector<MObject> drivers = {
			//NodeT::aStGraph,
			//NodeT::aStInput,
			//NodeT::aStOpName
			aStGraph,
			aStInput,
			aStOpName
			//aStParam,
			//aStElData
		};
		driversVec.insert(driversVec.end(), drivers.begin(), drivers.end());

		std::vector<MObject> driven = {
			//NodeT::aStOutput
			aStOutput,
			aStOpNameOut
		};
		drivenVec.insert(drivenVec.end(), driven.begin(), driven.end());

		std::vector<MObject> toAdd = {
			//NodeT::aStOutput
			aStGraph, aStInput, aStOpName, 
			aStOpNameOut, aStOutput,
		};
		toAddVec.insert(toAddVec.end(), toAdd.begin(), toAdd.end());

		return s;
	}

	MStatus syncOpNameOut(MObject& nodeObj, MDataBlock& pData) {
		MS s;
		auto nameStr = getOpNameFromNode(nodeObj);
		MDataHandle nameHdl = pData.outputValue(aStOpNameOut, &s);
		MCHECK(s, "could not retrieve opNameOut handle to sync name");
		nameHdl.setString(nameStr.c_str());
		return MS::kSuccess;
	}

	MStatus compute(MObject& nodeObj, const MPlug& plug, MDataBlock& data);

	static MStatus legalConnection(
		//MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	);

	virtual MStatus connectionMade(
		MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	virtual MStatus connectionBroken(
		MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);



};


template <typename StrataOpT> 
struct StrataOpNodeTemplate : public StrataOpNodeBase {
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

	using thisStrataOpT = StrataOpT;

	//constexpr typename StrataOpT = ed::StrataOp;

	template<typename NodeT>
	StrataOpT* getStrataOp(const MObject& nodeObj) {
		// return pointer to the current op, in this node's graph
		//ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(strataOpIndex);
		ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(getOpIndexFromNode<NodeT>(nodeObj));
		if (nodePtr == nullptr) {
			return nullptr;
		}
		return static_cast<StrataOpT*>(nodePtr);
	}

	StrataOpT* getStrataOp(const MObject& nodeObj) {
		// return pointer to the current op, in this node's graph
		// a strcmp slower than the templated version
		ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(getOpIndexFromNode(nodeObj));
		if (nodePtr == nullptr) {
			return nullptr;
		}
		return static_cast<StrataOpT*>(nodePtr);
	}

	MStatus createNewOp(MObject& mayaNodeMObject, StrataOpT*& opPtrOut) {
		/* create new op, and if we have incoming nodes, make new connections to it
		*/
		MS s(MS::kSuccess);
		opPtrOut = nullptr;
		StrataOpT* opPtr = opGraphPtr.get()->addNode<StrataOpT>(
			getOpNameFromNode(mayaNodeMObject)
		);
		opPtrOut = opPtr;
		// check input connections
		MFnDependencyNode depFn(mayaNodeMObject );
		MPlug inPlug = depFn.findPlug(aStInput, true, &s);
		MCHECK(s, "ERROR creating new op, could not find networked aStInput plug");


		// get indices of each input op, set as node inputs
		for (unsigned int i = 0; i < inPlug.evaluateNumElements(&s); i++) {
			MCHECK(s, "ERROR in evaluateNumElements for strata input plug");
			opPtrOut->inputs.push_back(inPlug.elementByPhysicalIndex(i).asInt());
		}

		return s;
	}

	//virtual void setFreshGraph();


	//static void testNoTemplateFn();

	//template<typename NodeT>
	//static void testFn();
	


	// probably a way to get access to the attr MObjects in this mixin's scope, 
	// but by contrast I actually understand this way with the template
	template <typename T>
	static void setOpIndexOnMayaNode(int opIndex, MObject& thisNode) {
		// update the maya attribute to this struct's op index
		MFnDependencyNode thisFn(thisNode);
		MPlug opIndexPlug = thisFn.findPlug(thisNode, T::aStOutput, false);
		opIndexPlug.setInt(opIndex);
	}

	Status syncOp(StrataOpT* op, MDataBlock& data) {
		/* update op from maya node datablock -
		this should be good enough, updating raw from MObject and plugs
		seems asking for trouble

		also set topoDirty / dataDirty flags here

		can't run this on newly created op directly, need to wait for compute
		*/
		return Status();
	}

	MStatus syncOpInputs(StrataOpT* op, const MObject& node);

	// shared compute function for all op nodes
	MStatus compute(MObject& thisMObj, const MPlug& plug, MDataBlock& data) {
		return StrataOpNodeBase::compute(
			thisMObj, plug, data
		);
	}

	void onInputConnectionChanged(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc);

	virtual MStatus setFreshGraph(MObject& nodeObj) {
		// make new internal graph, and also add this node's op to it
		DEBUGSL("template setFreshGraph")
		MS s = StrataOpNodeBase::setFreshGraph(nodeObj);
		MCHECK(s, "Error setting fresh graph, halting before adding op");
		thisStrataOpT* opOutPtr;
		s = createNewOp(nodeObj, opOutPtr);
		MCHECK(s, "Error adding new op to graph");
		DEBUGSL("created new op, added to graph");

		return s;

	}

	void postConstructor(MObject& nodeObj) {
		StrataOpNodeBase::postConstructor(nodeObj);
	}

	MStatus syncIncomingGraphData(MObject& nodeObj);


};


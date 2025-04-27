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

# define STRATABASE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStGraph; \
prefix MObject nodeT aStInput; \
prefix MObject nodeT aStOpName; \
prefix MObject nodeT aStOpNameOut; \
prefix MObject nodeT aStOutput; \


/// after all, why not
/// why shouldn't we inherit a base class from MPxNode
//struct StrataOpNodeBase : public MPxNode {

// here we template the base class, so each Maya node is specialised for a certain kind of Strata op?
// TEMPLATING BASE CLASS CAUSED GREAT PAIN
// now base class is real, an INTERMEDIATE class is templated, and that gets inherited into final Maya node class, alongside MPxNode etc

/* we pass a MObject& nodeObj param to all methods in the abstract bases,
to avoid any ambiguity with the proper MPxNode methods
*/

// you CANNOT repeat CANNOT share a base class attr MObject among multiple node classes.
// that really wasn't surprising but now we know for sure.

/* Q: why are the templates set up this way / why not put everything in one base / why isn't it correct - 
	A: because I really just want this to work. we can make it correct once it exists
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

	// index to use for output - cached here for access without maya eval
	int outputOpIndex = -1;


	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);

	template <typename NodeT>
	static const std::string getOpNameFromNode(MObject& nodeObj) {
		/* return a default name for strata op -
		if nothing defined in string field, use name of node itself*/
		//DEBUGS("GET OP NAME")
		//NODELOGT(NodeT, "GET OP NAME");
		MFnDependencyNode depFn(nodeObj);
		MStatus s;
		MPlug opNameFieldPlug = depFn.findPlug(NodeT::aStOpName, false, &s);
		if (s.error()) {
			DEBUGS("error getting op name field for node " + depFn.name());
			return "";
		}
		if (opNameFieldPlug.asString() != "") {
			return opNameFieldPlug.asString().asChar();
		}
		return depFn.name().asChar();
	}

	template <typename NodeT>
	static const std::string getOpNameFromNode(MDataBlock& data) {
		/* return a default name for strata op -
		if nothing defined in string field, use name of node itself*/
		MStatus s;
		if (data.inputValue(NodeT::aStOpName).asString().isEmpty()) {
			MFnDependencyNode depFn(NodeT::thisMObject());
			return depFn.name().asChar();
		}
		//if (opNameFieldPlug.asString() != "") {
		return data.inputValue(NodeT::aStOpName).asString();
		//}
		
	}

	template<typename NodeT>
	static inline const int getOpIndexFromNode(MObject& nodeObj) {
		/* return a default name for strata op -
		if nothing defined in string field, use name of node itself*/
		MFnDependencyNode depFn(nodeObj);
		MStatus s;
		MPlug opNameFieldPlug = depFn.findPlug(NodeT::aStOutput, false, &s);
		if (s.error()) {
			DEBUGS("error getting op index field for node " + depFn.name());
			return -1;
		}

		return opNameFieldPlug.asInt(); // NB - this might cause loops - if so, don't put it in driven.
	};
	template<typename NodeT>
	static inline const int getOpIndexFromNode(MDataBlock& data) {
		return data.outputValue(NodeT::aStOutput).asInt();
	}
	template<typename NodeT>
	static MStatus setOpIndexOnNode(MObject& nodeObj, int index) {
		MStatus s;
		if (nodeObj.isNull()) {
			//STAT_ERROR(s, "setOpIndex nodeObj is null");
			return MS::kFailure;
		}
		MFnDependencyNode nodeFn(nodeObj);
		nodeFn.findPlug(NodeT::aStOutput, false).setInt(index);
		return s;
	}

	template<typename NodeT>
	static MStatus setOpIndexOnNode(MDataBlock& data, int index) {
		MStatus s;
		data.outputValue(NodeT::aStOutput).setInt(-1);
		data.outputValue(NodeT::aStOutput).setInt(index);
		return s;
	}

	template<typename NodeT>
	static MStatus cacheOpIndexOnNodeObject(MObject& nodeObj, int index) {
		MStatus s;
		if (nodeObj.isNull()) {
			DEBUGSL("cacheOpIndex nodeObj is null");
			return MS::kFailure;
		}
		MFnDependencyNode nodeFn(nodeObj);
		NodeT* userPtr = dynamic_cast<NodeT*>(nodeFn.userNode());
		if (userPtr == nullptr) {
			//STAT_ERROR(s, "USERPTR in dynamic cast is null");
			DEBUGSL("USERPTR in dynamic cast is null");
			return MS::kFailure;
		}
		userPtr->outputOpIndex = index;
		return s;
	}

	MStatus setFreshGraph(MObject& nodeObj);
	MStatus setFreshGraph(MObject& nodeObj, MDataBlock& data);

	template<typename NodeT>
	MStatus syncIncomingGraphConnections(
		MObject& nodeObj) {
		/* update held pointers to connected nodes -
		can be slow, only happens while network is built.

		first check that we have a graph connection,
		then if we have any inputs -

		for purposes of merging separate graph flows we also copy pointers to the
		graphs -
		however usually it is enough to supply the op indices to op inputs (if
		we're working within only one strata graph) and the rest will be taken care of
		*/
		MS s = MS::kSuccess;
		MFnDependencyNode depFn(nodeObj);
		MPlug graphInPlug = depFn.findPlug(NodeT::aStGraph, true, &s);
		MCHECK(s, "Could not find plug for aStGraph for node" + depFn.name());
		if (graphInPlug.isConnected()) { // cast the input node to StrataNodeBase
			StrataOpNodeBase* nodePtr;
			s = castToUserNode<StrataOpNodeBase>(graphInPlug.node(), nodePtr);
			MCHECK(s, "Could not cast driving node " + MFnDependencyNode(graphInPlug.node()).name());
			sourceGraphPtr = nodePtr->opGraphPtr;
		}

		// copy any input graphs
		MPlug inputPlug = depFn.findPlug("stInput", true, &s);
		MCHECK(s, "Could not get stInput plug for node " + depFn.name() + " , cannot sync graph connections, halting");
		incomingGraphPtrs.clear();

		for (unsigned int i = 0; i < inputPlug.numElements(&s); i++) {
			MCHECK(s, "Error iterating numConnectedElements after reserve on stInput, i:" + std::to_string(i));
			MPlug inPlug = inputPlug.elementByPhysicalIndex(i);
			if (inPlug.isNull()) {
				continue;
			}
			if (inPlug.isConnected()) {
				// look up incoming graph
				MObject otherNodeObj = inPlug.source().node();
				StrataOpNodeBase* otherNodePtr;
				MStatus castS = castToUserNode(otherNodeObj, otherNodePtr);
				if (!castS) {
					DEBUGS("could not cast incoming node " + inPlug.source().name() + " to StrataOpNodeBase");
					continue;
				}
				// get graph from node
				incomingGraphPtrs.insert(
					std::make_pair(inPlug.logicalIndex(),
						//	std::move(
						std::weak_ptr<StrataOpGraph>(otherNodePtr->opGraphPtr)
						//)
					)
				);
			}
		}
		return s;
	};

	template<typename NodeT>
	MStatus syncIncomingGraphData(MObject& nodeObj, MDataBlock& data) {
		/* copy incoming graph data - extend to also add new node into graph
		* check through incoming map - if 0 entry found, copy it to a new object
		*/
		DEBUGS("base sync incoming")
			MS s = MS::kSuccess;
		if (incomingGraphPtrs.find(0) == incomingGraphPtrs.end()) {
			// 0 not found, make new graph
			//s = setFreshGraph(nodeObj);
			s = this->setFreshGraph(nodeObj);
			return s;
		}
		if (incomingGraphPtrs[0].expired()) {
			// 0 weak pointer expired (somehow), make new graph
			//setFreshGraph(nodeObj);
			this->setFreshGraph(nodeObj);
			return s;
		}
		opGraphPtr = incomingGraphPtrs[0].lock().get()->cloneShared<StrataOpGraph>();
		return s;
	}

	void postConstructor(MObject& nodeObj) {
		/* ensure graph pointer is reset*/
		DEBUGS("Base postConstructor")
			addedToGraph = true;
		//return;
		MS s = setFreshGraph(nodeObj);
	};

	static int strataAttrsDefined;

	// override to update node parametres
	// check yourself if params have changed, exps need to be recompiled etc
	virtual MStatus syncStrataParams(MObject& nodeObj, MDataBlock& data) {
		return MS::kSuccess;
	}

	template<typename NodeT>
	static MStatus defineStrataAttrs() {

		MS s(MS::kSuccess);

		MFnNumericAttribute nFn;
		MFnCompoundAttribute cFn;
		MFnTypedAttribute tFn;

		// all nodes connected directly to master graph node
		NodeT::aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kInt);
		nFn.setReadable(false);
		nFn.setStorable(false);
		nFn.setChannelBox(false);
		nFn.setKeyable(false);

		NodeT::aStInput = nFn.create("stInput", "stInput", MFnNumericData::kInt);
		nFn.setReadable(false);
		nFn.setArray(true);
		nFn.setUsesArrayDataBuilder(true);
		nFn.setIndexMatters(true);
		nFn.setDefault(-1);
		// strata inputs use physical order always - there's no reason ever to have an empty entry here
		// YES THERE IS NB - some nodes might only want aux inputs, but still create a fresh geo stream
		//nFn.setDisconnectBehavior(MFnAttribute::kDelete);
		nFn.setDisconnectBehavior(MFnAttribute::kReset);

		NodeT::aStOpName = tFn.create("stOpName", "stOpName", MFnData::kString);
		tFn.setDefault(MFnStringData().create(""));
		
		NodeT::aStOpNameOut = tFn.create("stOpNameOut", "stOpNameOut", MFnData::kString);
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
		NodeT::aStOutput = nFn.create("stOutput", "stOutput", MFnNumericData::kInt, -1);
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

	template<typename NodeT>
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
			s = defineStrataAttrs<NodeT>();
			MCHECK(s, "error DEFINING strata attrs");
			//strataAttrsDefined = 1;
		}

		std::vector<MObject> drivers = {
			//NodeT::aStGraph,
			//NodeT::aStInput,
			//NodeT::aStOpName
			NodeT::aStGraph,
			NodeT::aStInput,
			NodeT::aStOpName
			//aStParam,
			//aStElData
		};
		driversVec.insert(driversVec.end(), drivers.begin(), drivers.end());

		std::vector<MObject> driven = {
			//NodeT::aStOutput
			NodeT::aStOutput,
			NodeT::aStOpNameOut
		};
		drivenVec.insert(drivenVec.end(), driven.begin(), driven.end());

		std::vector<MObject> toAdd = {
			//NodeT::aStOutput
			NodeT::aStGraph, NodeT::aStInput, NodeT::aStOpName,
			NodeT::aStOpNameOut, NodeT::aStOutput,
		};
		toAddVec.insert(toAddVec.end(), toAdd.begin(), toAdd.end());

		return s;
	}

	template<typename NodeT>
	MStatus syncOpNameOut(MObject& nodeObj, MDataBlock& pData) {
		MS s;
		auto nameStr = getOpNameFromNode<NodeT>(nodeObj);
		MDataHandle nameHdl = pData.outputValue(NodeT::aStOpNameOut, &s);
		MCHECK(s, "could not retrieve opNameOut handle to sync name");
		nameHdl.setString(nameStr.c_str());
		return MS::kSuccess;
	}

	template <typename NodeT>
	MStatus compute(MObject& nodeObj, const MPlug& plug, MDataBlock& data) {
		MS s(MS::kSuccess);
		// sync op name
		if (plug.attribute() == NodeT::aStOpNameOut) {
			syncOpNameOut<NodeT>(nodeObj, data);
			data.setClean(plug);
			return s;
		}

		// copy graph data if it's dirty
		if (!data.isClean(NodeT::aStGraph)) {
			s = syncIncomingGraphData<NodeT>(nodeObj, data);
			DEBUGS("syncIncoming graph complete")
				MCHECK(s, "error syncing incoming graph data");
		}
		s = syncStrataParams(nodeObj, data);
		DEBUGSL("base synced strata params")
			MCHECK(s, "error syncing strata params");

		Status graphS;
		int upToNode = getOpIndexFromNode<NodeT>(data);
		DEBUGSL("base got index from node")
			opGraphPtr.get()->evalGraph(graphS, upToNode);
		DEBUGSL("base graph eval'd");

		data.setClean(NodeT::aStOutput);
		data.setClean(NodeT::aStOpName);

		return s;
	}

	template <typename NodeT>
	static MStatus legalConnection(
		//MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	) {
		// check that an input to the array only comes from another strata maya node - 
		// can't just connect a random integer
		DEBUGS("Base legalConnection")

		if (plug.attribute() == NodeT::aStGraph) { // main graph input
			if (otherPlug.node() == plug.node()) { // no feedback loops
				isLegal = false;
				return MS::kSuccess;
			}
			if (MFnAttribute(otherPlug.attribute()).name() == "stOutput") {
				isLegal = true;
				return MS::kSuccess;
			}
			isLegal = false; // only allow graph connections from graph outputs
			return MS::kSuccess;
		}
		
		// stInput can just be integers I think? we can check more carefully for copying input graphs when they change

		//if (MFnAttribute(plug.attribute()).name() == "stInput") {
		//	if (otherPlug.node() == plug.node()) { // no feedback loops
		//		isLegal = false;
		//		return MS::kSuccess;
		//	}
		//	if (MFnAttribute(otherPlug.attribute()).name() == "stOutput") {
		//		isLegal = true;
		//		return MS::kSuccess;
		//	}
		//	isLegal = false;
		//	return MS::kSuccess;
		//}
		return MStatus::kUnknownParameter;
	}

	template <typename NodeT>
	MStatus connectionMade(
		MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	) {
		/* on connection made from another strata op node,
		copy the graph of the incoming node
		*/
		MStatus s = MS::kSuccess;

		DEBUGS("CONNECTION MADE begin");
		if (!addedToGraph) {
			DEBUGS("node not yet added to graph, skipping connection made");
			return s;
		}


		DEBUGS("other plug name" + plug.name());

		DEBUGS("plug is null" + std::to_string(plug.isNull()));
		DEBUGS("plug name" + plug.name());
		DEBUGS("other plug name" + plug.name());

		if (MFnAttribute(plug.attribute()).name() != "stInput") {// crash
			return s;
		}

		/*StrataOpNodeBase* otherNodePtr;
		s = castToUserNode<StrataOpNodeBase>(otherPlug.node(), otherNodePtr);*/
		MCHECK(s, "ERROR casting incoming strata connection to StrataOpNodeBase");
		//s = refreshGraphPtr(otherNodePtr->opGraphPtr.get());
		s = syncIncomingGraphConnections<NodeT>(plug.node());
		MCHECK(s, "ERROR refreshing graph pointer from incoming node connection");
		return s;
	}
	template <typename NodeT>
	MStatus connectionBroken(
		MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	) {
		MStatus s = MS::kSuccess;
		//if (plug.attribute() != aStInput) {
		if (MFnAttribute(plug.attribute()).name() != "stInput") {
			return s;
		}
		//refreshGraphPtr(nullptr);
		syncIncomingGraphConnections<NodeT>(plug.node());
		return s;
	}


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
//
//	// pointer to this node's op
//// retrieve anytime the node or graph changes
//	StrataOpT* opPtr = nullptr;

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

	template<typename NodeT>
	StrataOpT* getStrataOp(MDataBlock& data) {
		// return pointer to the current op, in this node's graph
		// a strcmp slower than the templated version
		if (opGraphPtr.get() == nullptr) {
			DEBUGSL("no graph pointer initialised, cannot get strata op");
			return nullptr;
		}
		int opIndex = getOpIndexFromNode<NodeT>(data);
		ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(opIndex);
		if (nodePtr == nullptr) {
			DEBUGSL("unable to get node pointer - index from node is " + std::to_string(opIndex));
			return nullptr;
		}
		return static_cast<StrataOpT*>(nodePtr);
	}

	template<typename NodeT>
	MStatus createNewOp(MObject& mayaNodeMObject, MDataBlock& data, StrataOpT*& opPtrOut) {
		/* create new op, and if we have incoming nodes, make new connections to it
		* 
		* I KNOW THE TEMPLATING HERE SUCKS
		* we have to template this with the final real class
		* i'll fix it someday i swear
		*/
		DEBUGS("createNewOp")
		MS s(MS::kSuccess);
		opPtrOut = nullptr;
		StrataOpT* opPtr = opGraphPtr.get()->addNode<StrataOpT>(
			data.outputValue(NodeT::aStOpNameOut).asString().asChar()
		);
		opPtrOut = opPtr;
		DEBUGSL("added new op to graph")
		// set op index on node output
		s = setOpIndexOnNode<NodeT>(data, opPtr->index);
		MCHECK(s, "ERROR creating new op, could not set op index on node");
		s = cacheOpIndexOnNodeObject<NodeT>(mayaNodeMObject, opPtr->index);
		MCHECK(s, "ERROR creating new op, could not cache op index on node");

		//// set new node as the graph output - caveman solution to subtle situation
		opGraphPtr.get()->outNodeIndex = opPtr->index;

		//DEBUGS("set op index on node")
		// check input connections

		MArrayDataHandle arrDh = data.inputArrayValue(NodeT::aStInput);
		for (unsigned int i = 0; i < arrDh.elementCount(); i++) {
			opPtrOut->inputs.push_back(arrDh.inputValue().asInt());
			arrDh.next();
		}
		DEBUGS("pulled op inputs");



		return s;
	}

	template<typename NodeT>
	MStatus createNewOp(MObject& mayaNodeMObject, StrataOpT*& opPtrOut) {
		/* create new op, and if we have incoming nodes, make new connections to it
		* this is only called on postConstructor to set up first op when node is created
		*/
		DEBUGS("createNewOp new data");
		MS s(MS::kSuccess);
		opPtrOut = nullptr;
		StrataOpT* opPtr = opGraphPtr.get()->addNode<StrataOpT>(
			//data.outputValue(NodeT::aStOpNameOut).asString().asChar()
			getOpNameFromNode<NodeT>(mayaNodeMObject)
		);
		opPtrOut = opPtr;
		DEBUGSL("added new op to graph")
			// set op index on node output
			s = setOpIndexOnNode<NodeT>(mayaNodeMObject, opPtr->index);
			s = cacheOpIndexOnNodeObject<NodeT>(mayaNodeMObject, opPtr->index);
		DEBUGS("set op index on node")
			// check input connections
			/*MFnDependencyNode depFn(mayaNodeMObject );
			MPlug inPlug = depFn.findPlug(aStInput, true, &s);*/
			MCHECK(s, "ERROR creating new op, could not find networked aStInput plug");
		//MArrayDataHandle arrDh = data.inputArrayValue(NodeT::aStInput);
		//for (unsigned int i = 0; i < arrDh.elementCount(); i++) {
		//	opPtrOut->inputs.push_back(arrDh.inputValue().asInt());
		//	arrDh.next();
		//}
		//DEBUGS("pulled op inputs")

			//// get indices of each input op, set as node inputs
			//for (unsigned int i = 0; i < inPlug.evaluateNumElements(&s); i++) {
			//	MCHECK(s, "ERROR in evaluateNumElements for strata input plug");
			//	opPtrOut->inputs.push_back(inPlug.elementByPhysicalIndex(i).asInt());
			//}

			return s;
	}



	//static void testNoTemplateFn();

	//template<typename NodeT>
	//static void testFn();
	

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



	void onInputConnectionChanged(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc);

	template<typename NodeT>
	MStatus setFreshGraph(MObject& nodeObj, MDataBlock& data) {
		// make new internal graph, and also add this node's op to it
		DEBUGSL("template setFreshGraph")
		MS s = StrataOpNodeBase::setFreshGraph(nodeObj);
		MCHECK(s, "Error setting fresh graph, halting before adding op");
		StrataOpT* opOutPtr;
		s = createNewOp(nodeObj, data, opOutPtr);
		MCHECK(s, "Error adding new op to graph");
		DEBUGSL("created new op, added to graph");
		return s;
	}

	template<typename NodeT>
	MStatus setFreshGraph(MObject& nodeObj) {
		// make new internal graph, and also add this node's op to it
		DEBUGSL("template setFreshGraph")
			MS s = StrataOpNodeBase::setFreshGraph(nodeObj);
		MCHECK(s, "Error setting fresh graph, halting before adding op");
		thisStrataOpT* opOutPtr;
		s = createNewOp<NodeT>(nodeObj, opOutPtr);
		MCHECK(s, "Error adding new op to graph");
		DEBUGSL("created new op, added to graph");
		return s;
	}

	template<typename NodeT>
	void postConstructor(MObject& nodeObj) {
		//StrataOpNodeBase::postConstructor(nodeObj);
		DEBUGS("template postConstructor")
			addedToGraph = true;
		//return;
		MS s = setFreshGraph<NodeT>(nodeObj);
	}

	template<typename NodeT>
	MStatus syncIncomingGraphData(MObject& nodeObj, MDataBlock& data) {
		/* copy graph, and add new node to it*/
		MStatus s = StrataOpNodeBase::syncIncomingGraphData<NodeT>(nodeObj, data);
		DEBUGS("template syncIncoming")
			MCHECK(s, "Error copying graph data from OpNodeBase, halting");
		StrataOpT* opPtr;
		s = createNewOp<NodeT>(nodeObj, data, opPtr);
		MCHECK(s, "Error adding new op to new graph");
		return s;
	};

	template <typename NodeT>
	MStatus compute(MObject& nodeObj, const MPlug& plug, MDataBlock& data) {
		MS s(MS::kSuccess);
		// sync op name
		if (plug.attribute() == NodeT::aStOpNameOut) {
			syncOpNameOut<NodeT>(nodeObj, data);
			data.setClean(plug);
			return s;
		}

		// copy graph data if it's dirty
		if (!data.isClean(NodeT::aStGraph)) {
			s = syncIncomingGraphData<NodeT>(nodeObj, data);
			DEBUGS("syncIncoming graph complete")
				MCHECK(s, "error syncing incoming graph data");
		}
		s = syncStrataParams(nodeObj, data);
		DEBUGSL("base synced strata params")
			MCHECK(s, "error syncing strata params");

		Status graphS;
		int upToNode = getOpIndexFromNode<NodeT>(data);
		//DEBUGSL("template got index from node")
		opGraphPtr.get()->evalGraph(graphS, upToNode);
		DEBUGSL("template graph eval'd");

		data.setClean(NodeT::aStOutput);
		data.setClean(NodeT::aStOpName);

		return s;
	}
};


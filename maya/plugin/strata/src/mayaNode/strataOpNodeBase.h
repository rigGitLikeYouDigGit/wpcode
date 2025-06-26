#pragma once

#include <cstdlib>

#include <maya/MObject.h>
#include "../MInclude.h"
#include "../stratacore/op.h"
#include "../stratacore/opgraph.h"
#include "../logger.h"

//#include "../strataop/elementOp.h" // TEMP


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


/*
Q: NAMING
take the node name, unless the plug has a string set on it
what if the node name is changed? there's no event for it
	A: then we don't know or care until the next node compute. If you want your graph to update, 
		make the node recompute for something. it's not hard

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

	// cache op name on node so we don't have to do random computes to get strata ops
	std::string cachedNamePlugValue = "";
	

	// semantic/logical plug index to incoming graph
	std::map<int, std::weak_ptr<ed::StrataOpGraph>> incomingGraphPtrs = {};
	// weak anyway, just in case
	/* NB this means we'll have to merge multiple incoming graphs when we get to using
	multiple inputs - probably fine*/

	// index to use for output - cached here for access without maya eval
	int outputOpIndex = -1;


	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);

	std::string getOpNameFromNode(MObject nodeObj) {
		/* no dg eval, rely on compute to update cached name value*/
		if (cachedNamePlugValue.empty()) {
			return std::string(MFnDependencyNode(nodeObj).name().asChar());
		}
		return cachedNamePlugValue;
	}

	//template <typename NodeT>
	//static const std::string getOpNameFromNode(MObject& nodeObj) {
	//	/* return a default name for strata op -
	//	if nothing defined in string field, use name of node itself*/
	//	//DEBUGS("GET OP NAME")
	//	//NODELOGT(NodeT, "GET OP NAME");
	//	MFnDependencyNode depFn(nodeObj);
	//	MStatus s;
	//	MPlug opNameFieldPlug = depFn.findPlug(NodeT::aStOpName, false, &s);
	//	if (s.error()) {
	//		DEBUGS("error getting op name field for node " + depFn.name());
	//		return "";
	//	}
	//	if (opNameFieldPlug.asString() != "") {
	//		return opNameFieldPlug.asString().asChar();
	//	}
	//	return depFn.name().asChar();
	//}

	//template <typename NodeT>
	//static const std::string getOpNameFromNode(MDataBlock& data) {
	//	/* return a default name for strata op -
	//	if nothing defined in string field, use name of node itself*/
	//	MStatus s;
	//	if (data.inputValue(NodeT::aStOpName).asString().isEmpty()) {
	//		MFnDependencyNode depFn(NodeT::thisMObject());
	//		return depFn.name().asChar();
	//	}
	//	return data.inputValue(NodeT::aStOpName).asString();
	//}

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
		// THIS WILL TRIGGER COMPUTE IF DIRTY
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
		//data.outputValue(NodeT::aStOutput).setInt(-1);
		data.outputValue(NodeT::aStOutput).setInt(index);
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
		LOG(depFn.name() + ": SYNC INCOMING CONNECTIONS ");
		
		// copy any input graphs
		MPlug inputPlug = depFn.findPlug("stInput", true, &s);
		//MPlug inputPlug = depFn.findPlug("stInput", false, &s); 
		/* wantNetworkedPlug
		* truly one of life's greatest mysteries
		*/
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
				MStatus castS = castToUserNode<StrataOpNodeBase>(otherNodeObj, otherNodePtr);
				if (!castS) {
					l("could not cast incoming node " + inPlug.source().name() + " to StrataOpNodeBase");
					continue;
				}
				// get graph from node
				incomingGraphPtrs.insert(
					std::make_pair(inPlug.logicalIndex(),
						//	std::move(
						std::weak_ptr<ed::StrataOpGraph>(otherNodePtr->opGraphPtr)
						//)
					)
				);
			}
		}
		return s;
	};


	template<typename NodeT>
	MStatus syncIncomingGraphData(MObject& nodeObj, MDataBlock& data) {
		return syncIncomingGraphData<NodeT>(nodeObj);
	}

	template<typename NodeT>
	MStatus syncIncomingGraphData(MObject& nodeObj) {
		/* copy incoming graph data - extend to also add new node into graph
		* check through incoming map - if 0 entry found, copy it to a new object
		*/
		Status strataS;
		LOG(MFnDependencyNode(nodeObj).name() + ": SYNC INCOMING DATA" );
			MS s = MS::kSuccess;
		if (incomingGraphPtrs.find(0) == incomingGraphPtrs.end()) {
			// 0 not found, make new graph
			//s = setFreshGraph(nodeObj);
			l("no input 0 found - creating fresh graph");
			s = this->setFreshGraph(nodeObj);
			//return s;
		}
		else if (incomingGraphPtrs[0].expired()) {
			// 0 weak pointer expired (somehow), make new graph
			//setFreshGraph(nodeObj);			DEBUGS("no input 0 found - creating fresh graph");

			l("input 0 expired - creating fresh graph");
			this->setFreshGraph(nodeObj);
			//return s;
		}
		else {
			l("cloning graph from input 0");
			opGraphPtr = incomingGraphPtrs[0].lock().get()->cloneShared<ed::StrataOpGraph>(false);
		}

		// now merge all graphs from inputs
		// TODO: optimise this, cache on this node a hash value or an op counter for each input or something
		for (auto& pair : incomingGraphPtrs) {
			if (pair.first == 0) {
				continue;
			}
			if (pair.second.expired()) {
				continue;
			}
			l("merging input graph " + str(pair.first));
			opGraphPtr.get()->mergeOther(*pair.second.lock().get(), false, strataS);
			//CWSTAT(strataS, "Error merging incoming graph to input " + std::to_string(pair.first));
			CWMSG(strataS, "Error merging incoming graph to input " + std::to_string(pair.first));
		}
		return s;
	}

	void postConstructor(MObject& nodeObj) {
		/* ensure graph pointer is reset*/
		incomingGraphPtrs = std::map<int, std::weak_ptr<ed::StrataOpGraph>>{};
		LOG(MFnDependencyNode(nodeObj).name() + ": Base postConstructor");
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
		nFn.setAffectsAppearance(true);
		//nFn.setAffectsWorldSpace(true);

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

	static std::string nodeName(MObject obj) {
		return std::string(MFnDependencyNode(obj).name().asChar());
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
			//NodeT::aStGraph,
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
			//NodeT::aStGraph, 
			NodeT::aStInput, NodeT::aStOpName,
			NodeT::aStOpNameOut, NodeT::aStOutput,
		};
		toAddVec.insert(toAddVec.end(), toAdd.begin(), toAdd.end());

		return s;
	}

	template<typename NodeT>
	MStatus syncOpNameOut(MObject& nodeObj, MDataBlock& pData) {
		MS s;
		cachedNamePlugValue = std::string(pData.inputValue(NodeT::aStOpName).asString().asChar());
		std::string nameStr = getOpNameFromNode(nodeObj);
		MDataHandle nameHdl = pData.outputValue(NodeT::aStOpNameOut, &s);
		MCHECK(s, "could not retrieve opNameOut handle to sync name");
		nameHdl.setString(nameStr.c_str());
		return MS::kSuccess;
	}

	template <typename NodeT>
	MStatus compute(MObject& nodeObj, const MPlug& plug, MDataBlock& data) {
		/* in the case of the name, it would help for this function
		to be able to say, "computation is completely done, exit immediately"
		maybe in general we return kUnknown, and kSuccess if completely done
		*/
		MS s(MS::kSuccess);
		// sync op name
		if ((!data.isClean(NodeT::aStName)) || (plug.attribute() == NodeT::aStOpNameOut)) {
			syncOpNameOut<NodeT>(nodeObj, data);
			data.setClean(NodeT::aStName);
			if (plug.attribute() == NodeT::aStOpNameOut) {
				return MS::kEndOfFile;
			}
		}


		LOG(nodeName(nodeObj) + " BASE compute");

		// copy graph data if it's dirty
		//if (!data.isClean(NodeT::aStGraph)) {
		if (!data.isClean(NodeT::aStInput)) {
			s = syncIncomingGraphData<NodeT>(nodeObj, data);
			l("syncIncoming graph complete");
				MCHECK(s, "error syncing incoming graph data");
		}
		s = syncStrataParams(nodeObj, data);
		l("base synced strata params");
			MCHECK(s, "error syncing strata params");

		Status graphS;
		int upToNode = getOpIndexFromNode<NodeT>(data);
		l("base got index from node");
			opGraphPtr.get()->evalGraph(graphS, upToNode);
		l("base graph eval'd");

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
		LOG(nodeName(plug.node()) + " legal connection");
		MFnDependencyNode depFn(plug.node());
		if (!asSrc) {
			DEBUGS(depFn.name() + " Base legalConnection from " + otherPlug.name() + " to " + plug.name());
		}
		else {
			DEBUGS(depFn.name() + "Base legalConnection from " + plug.name() + " to " + otherPlug.name());
		}

		/* for some bizarre reason on node creation I get calls trying to connect a node's
		message attribute to itself*/
		if (plug.node() == otherPlug.node()) {
			if (plug.attribute() == otherPlug.attribute()) {
				DEBUGSL("prevented self-connection on plug: " + plug.name());
				isLegal = false;
				return MS::kSuccess;
			}
		}

		if (plug.attribute() == NodeT::aStInput) { // main graph input
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
		MFnDependencyNode depFn(nodeObj);

		LOG("CONNECTION MADE begin");
		if (!addedToGraph) {
			l("node not yet added to graph, skipping connection made");
			return s;
		}

		if (asSrc) {
			l("from this " + plug.name() + " to other " + otherPlug.name());
		}
		else {
			l("from other " + otherPlug.name() + " to this " + plug.name());
		}

		// we only care about stInput
		if (MFnAttribute(plug.attribute()).name() != "stInput") {
			l("returning non-input connection");
			return s;
		}

		s = syncIncomingGraphConnections<NodeT>(plug.node());
		MCHECK(s, "ERROR refreshing graph pointer on connectionMade");
		//s = syncIncomingGraphData<NodeT>(plug.node());
		//MCHECK(s, "ERROR merging incoming graphs on connectionMade");
		/* actual operations to copy and merge graphs need to be done in compute,
		since if a structural change happens more than one maya node before this one,
		this function won't re-run*/
		return s;
	}
	template <typename NodeT>
	MStatus connectionBroken(
		MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	) {
		MFnDependencyNode depFn(nodeObj);
		DEBUGS("BREAK CONNECTION: " + depFn.name());
		MStatus s = MS::kSuccess;
		//if (plug.attribute() != aStInput) {
		if (MFnAttribute(plug.attribute()).name() != "stInput") {
			return s;
		}
		s = syncIncomingGraphConnections<NodeT>(plug.node());
		MCHECK(s, "ERROR refreshing graph pointer on connectionBroken");
		//s = syncIncomingGraphData<NodeT>(plug.node());
		MCHECK(s, "ERROR merging incoming graphs on connectionBroken");
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

	//template<typename NodeT>
	//StrataOpT* getStrataOp() {
	//	if (!opGraphPtr) {
	//		return nullptr;
	//	}
	//	if()
	//}

	template<typename NodeT>
	StrataOpT* getStrataOp(MObject& nodeObj) {
		// return pointer to the current op, in this node's graph
		// add graph if null
		/* consider lookup to user MPx node here to avoid forcing compute - 
		SEEMS better and less active for a simple getter function*/
		LOG("get strata op by obj");

		if (!opGraphPtr) {
			return nullptr;
		}
		return opGraphPtr->getNode<StrataOpT>(getOpNameFromNode(nodeObj));

		//if (!cachedOpName.empty()) {
		//	return opGraphPtr->getNode<StrataOpT>(opGraphPtr->nameIndexMap.at(cachedOpName));
		//}
		//int nodeIndex = getOpIndexFromNode<NodeT>(nodeObj);
		//ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(nodeIndex);
		//if (nodePtr == nullptr) {
		//	return nullptr;
		//}
		//return static_cast<StrataOpT*>(nodePtr);
	}

	//template<typename NodeT>
	//StrataOpT* getStrataOp(MDataBlock& data) {
	//	// return pointer to the current op, in this node's graph
	//	// a strcmp slower than the templated version
	//	LOG("get strata op by data");
	//	if (opGraphPtr.get() == nullptr) {
	//		l("no graph pointer initialised, cannot get strata op");
	//		return nullptr;
	//	}
	//	return opGraphPtr->getNode<StrataOpT>(getOpNameFromNode());
	//	
	//	if (!cachedOpName.empty()) {
	//		return opGraphPtr->getNode<StrataOpT>(opGraphPtr->nameIndexMap.at(cachedOpName));
	//	}
	//	else {

	//	}

	//	int opIndex = getOpIndexFromNode<NodeT>(data);
	//	ed::DirtyNode* nodePtr = opGraphPtr.get()->getNode(opIndex);
	//	if (nodePtr == nullptr) {
	//		l("unable to get node pointer - index from node is " + std::to_string(opIndex));
	//		return nullptr;
	//	}
	//	return static_cast<StrataOpT*>(nodePtr);
	//}

	template<typename NodeT>
	MStatus _createNewOpInner(MObject& mayaNodeMObject, StrataOpT*& opPtrOut, std::string opName) {
		LOG("_createNewOp inner");
		MS s(MS::kSuccess);

		// check existing
		auto nameLookup = opGraphPtr.get()->nameIndexMap.find(opName);
		if (nameLookup != opGraphPtr.get()->nameIndexMap.end()) {
			/* graph contains node of this name -
			if it's the output node (strata node from earlier node eval)
			then remove it,
			otherwise error, since we ban duplicate names

			NO this shouldn't be happening, graph should always be created fresh, there should be no holdover
			node left
			*/


			MGlobal::displayError(("strata op node " + (MFnDependencyNode(mayaNodeMObject).name()) + " tried to add a strata op with a name already found in incoming graph: " + MString(opName.c_str()) + " -- halting."));
			l("strata op node " + (MFnDependencyNode(mayaNodeMObject).name()) + " tried to add a strata op with a name already found in incoming graph: " + MString(opName.c_str()) + " -- halting.");
			return MS::kInvalidParameter;
		}

		opPtrOut = nullptr;
		StrataOpT* opPtr = opGraphPtr.get()->addNode<StrataOpT>(
			opName
		);
		opPtrOut = opPtr;
		l("added op: " + opPtr->name + ed::str(opPtr->index));

		// set graph's output index to this node
		opGraphPtr->setOutputNode(opPtr->index);

		// look up linked graphs, connect their output ops to this op's inputs
		std::vector<int> inputKeys = mapKeys(incomingGraphPtrs);
		int maxIndex = 0;
		if (inputKeys.size()) {
			maxIndex = *std::max_element(inputKeys.begin(), inputKeys.end());
			opPtr->inputs.reserve(maxIndex);
			for (int i = 0; i < static_cast<int>(incomingGraphPtrs.size()); i++) {
				l("check input graph index:" + ed::str(i));
				if (!incomingGraphPtrs.count(i)) {
					opPtr->inputs.push_back(-1);
					continue;
				}

				ed::StrataOpGraph* incomingGraphP = incomingGraphPtrs.at(i).lock().get();
				if (incomingGraphP == nullptr) {
					
					opPtr->inputs.push_back(-1);
					continue;
				}

				// check if incoming graph has no nodes
				if (!incomingGraphP->nodes.size()) {
					opPtr->inputs.push_back(-1);
					continue;
				}
				
				// check that name exists in map (should be impossible to fail here)
				if (opGraphPtr->nameIndexMap.find(incomingGraphP->outputNodeName())
					== opGraphPtr->nameIndexMap.end()
					) {
					l("graph outName: " + incomingGraphP->outputNodeName() + " missing from index map, seriously wrong");
					return MS::kFailure;
				}
				
				opPtr->inputs.push_back(
					opGraphPtr->nameIndexMap.at(
						incomingGraphP->outputNodeName()
					)
				);
			}
		}
		l("finished op inputs, return");

		return s;
	}

	template<typename NodeT>
	MStatus createNewOp(MObject& mayaNodeMObject, MDataBlock& data, StrataOpT*& opPtrOut) {
		/* create new op, and if we have incoming nodes, make new connections to it
		* 
		* I KNOW THE TEMPLATING HERE SUCKS
		* we have to template this with the final real class
		* i'll fix it someday i swear
		* 
		* check if op name already exists in graph - if yes, for now, ERROR
		* get it working then make it fancy
		*/
		LOG("createNewOp by data");
		MS s(MS::kSuccess);
		
		//std::string opName(data.outputValue(NodeT::aStOpNameOut).asString().asChar());

		s = _createNewOpInner<NodeT>(mayaNodeMObject, opPtrOut, getOpNameFromNode(mayaNodeMObject));

			l("added new op to graph");
		// set op index on node output
		s = setOpIndexOnNode<NodeT>(data, opPtrOut->index);
		MCHECK(s, "ERROR creating new op, could not set op index on node");
		
		return s;
	}

	//template<typename NodeT>
	//MStatus createNewOp(MObject& mayaNodeMObject, StrataOpT*& opPtrOut) {
	//	/* create new op, and if we have incoming nodes, make new connections to it
	//	*
	//	* I KNOW THE TEMPLATING HERE SUCKS
	//	* we have to template this with the final real class
	//	* i'll fix it someday i swear
	//	*
	//	* check if op name already exists in graph - if yes, for now, ERROR
	//	* get it working then make it fancy
	//	*/
	//	LOG("createNewOp by plug");
	//		MS s(MS::kSuccess);

	//	MFnDependencyNode depFn(mayaNodeMObject);
	//	std::string opName(depFn.findPlug(NodeT::aStOpNameOut, false).asString().asChar());
	//	s = _createNewOpInner<NodeT>(mayaNodeMObject, opPtrOut, opName);
	//	MCHECK(s, "ERROR on _createNewOpInner");
	//	if (opPtrOut == nullptr) {
	//		l("opPtr still null after creating new op for node" + opName);
	//		return s;
	//	}
	//	s = setOpIndexOnNode<NodeT>(mayaNodeMObject, opPtrOut->index);

	//	return s;
	//}

	template<typename NodeT>
	MStatus syncIncomingGraphData(MObject& nodeObj, MDataBlock& data) {

		/* copy incoming graph data - extend to also add new node into graph
		* check through incoming map - if 0 entry found, copy it to a new object
		*/
		MStatus result = StrataOpNodeBase::syncIncomingGraphData<NodeT>(nodeObj, data);
		StrataOpT* opPtr;
		result = createNewOp<NodeT>(nodeObj, data, opPtr);
		return result;
	}
	/* maybe this method should automatically add new op, as above -
	keeping it explicit in other functions for now?
	*/


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

	template <typename NodeT>
	MStatus connectionBroken(
		MObject& nodeObj,
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	) {
		MFnDependencyNode depFn(nodeObj);
		LOG("BREAK CONNECTION: " + depFn.name());
		MStatus s = MS::kSuccess;
		//if (plug.attribute() != aStInput) {
		if (MFnAttribute(plug.attribute()).name() != "stInput") {
			return s;
		}
		s = syncIncomingGraphConnections<NodeT>(plug.node());
		MCHECK(s, "ERROR refreshing graph pointer on connectionBroken");
		//s = syncIncomingGraphData<NodeT>(plug.node());
		MCHECK(s, "ERROR merging incoming graphs on connectionBroken");
		return s;
	}

	void onInputConnectionChanged(const MPlug& inputArrayPlug,
		const MPlug& otherPlug,
		bool 	asSrc);

	template<typename NodeT>
	MStatus setFreshGraph(MObject& nodeObj, MDataBlock& data) {
		// make new internal graph, and also add this node's op to it
		LOG("template setFreshGraph by data");
		MS s = StrataOpNodeBase::setFreshGraph(nodeObj, data);
		MCHECK(s, "Error setting fresh graph, halting before adding op");
		StrataOpT* opOutPtr;
		s = createNewOp<NodeT>(nodeObj, data, opOutPtr);
		MCHECK(s, "Error adding new op to graph");
		l("created new op, added to graph");
		return s;
	}

	//template<typename NodeT>
	//MStatus setFreshGraph(MObject& nodeObj) {
	//	// make new internal graph, and also add this node's op to it
	//	LOG("template setFreshGraph by obj");
	//		MS s = StrataOpNodeBase::setFreshGraph(nodeObj);
	//	MCHECK(s, "Error setting fresh graph, halting before adding op");
	//	thisStrataOpT* opOutPtr = nullptr;
	//	s = createNewOp<NodeT>(nodeObj, opOutPtr);
	//	MCHECK(s, "Error adding new op to graph");
	//	l("created new op, added to graph");
	//	return s;
	//}

	template<typename NodeT>
	void postConstructor(MObject& nodeObj) {
		/* ensure graph pointer is reset*/
		LOG("template postConstructor");
		StrataOpNodeBase::postConstructor(nodeObj);
		//return;

		//MS s = setFreshGraph<NodeT>(nodeObj);
	}


	template <typename NodeT>
	MStatus compute(MObject& nodeObj, const MPlug& plug, MDataBlock& data) {
		MS s(MS::kSuccess);
		MFnDependencyNode depFn(nodeObj);
		//ed::Log l(depFn.name() + ": st template compute ");
		LOG(depFn.name() + ": st template compute ");
		///* check first if op pointer is null - if yes, we're somehow computing before
		//running postConstructor*/
		// sync op name
		if ((!data.isClean(NodeT::aStOpName)) || (plug.attribute() == NodeT::aStOpNameOut)) {
			syncOpNameOut<NodeT>(nodeObj, data);
			data.setClean(NodeT::aStOpName);
			if (plug.attribute() == NodeT::aStOpNameOut) {
				return MS::kEndOfFile;
			}
		}


		StrataOpT* opPtr = getStrataOp<NodeT>(nodeObj); 
		if (opPtr == nullptr) {
			setFreshGraph<NodeT>(nodeObj, data);
		}
		// did that fix it?
		opPtr = getStrataOp<NodeT>(nodeObj);
		if (opPtr == nullptr) {
			/* man this program sucks*/
			l("opPtr is still null after setting fresh graph");
			return MS::kFailure;
		}

		

		// copy graph data if it's dirty
		//if (!data.isClean(NodeT::aStGraph)) {
		if (!data.isClean(NodeT::aStInput)) {
			s = syncIncomingGraphData<NodeT>(nodeObj, data);
			l("syncIncoming graph complete");
				MCHECK(s, "error syncing incoming graph data");

			//// add new op to graph, set as graph output
			//StrataOpT* opPtr = nullptr;
			//s = createNewOp<NodeT>(nodeObj, data, opPtr);
			//MCHECK(s, "failed to add new op to graph for node " )
		}
		s = syncStrataParams(nodeObj, data);
		l("template synced strata params");
			MCHECK(s, "error syncing strata params");

		Status graphS;
		int upToNode = getOpIndexFromNode<NodeT>(data);
		//DEBUGSL("template got index from node")
		graphS = opGraphPtr.get()->evalGraph(graphS, opGraphPtr->_outputIndex);
		CWSTAT(graphS);
		l("template graph eval'd");

		data.setClean(NodeT::aStOutput);
		data.setClean(NodeT::aStOpName);

		return s;
	}
};


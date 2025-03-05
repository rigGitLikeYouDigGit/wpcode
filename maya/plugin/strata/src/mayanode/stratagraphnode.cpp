
#pragma once

#include "strataGraphNode.h"

#include "../macro.h"
#include "../api.h"
#include "../lib.cpp"

using namespace ed;
MObject StrataGraphNode::aStGraph;
MObject StrataGraphNode::aStGraphName;

//MString StrataGraphNode::kNODE_NAME;
MString StrataGraphNode::kNODE_NAME = MString("strataGraph");
MTypeId StrataGraphNode::kNODE_ID = MTypeId(0x00122CA0);


MStatus StrataGraphNode::initialize() {
	MS s(MS::kSuccess);

	MFnTypedAttribute tFn;
	MFnNumericAttribute nFn;

	// name of graph object to create
	aStGraphName = tFn.create("stGraphName", "stGraphName", MFnData::kString);
	MFnStringData stringDataFn;
	tFn.setDefault(stringDataFn.create("newStrataGraph"));

	// pointer address to graph object in memory
	aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kBoolean);
	nFn.setWritable(false);
	nFn.setStorable(false);
	nFn.setKeyable(false);
	nFn.setChannelBox(false);

	std::vector<MObject> drivers = {
		aStGraphName
	};
	std::vector<MObject> driven = {
		aStGraph
	};
	addAttributes<StrataGraphNode>(drivers);
	addAttributes<StrataGraphNode>(driven);

	setAttributesAffect<StrataGraphNode>(drivers, driven);

	return s;
}

void StrataGraphNode::postConstructor() {
	/* initialise new strata op graph with this node's name*/
	//opGraph.reset(&StrataOpGraph());
	opGraph.reset(new ed::StrataOpGraph);
}

MStatus StrataGraphNode::compute(const MPlug& plug, MDataBlock& data) {
	// set the op graph's name, then set its memory address
	// first check if plug is clean
	MS s(MS::kSuccess);
	if (data.isClean(plug)) {
		return s;
	}

	// set name
	MDataHandle nameDH = data.inputValue(aStGraphName, &s);
	if (s != MS::kSuccess) {
		DEBUGS("COULD NOT GET GRAPH NAME DATA");
		return s;
	}

	MString plugVal = nameDH.asString();
	DEBUGS("plugName len: " + std::to_string(plugVal.length()));
	DEBUGS("plugName");
	DEBUGS(plugVal);

	int charLen;
	const char* nameChar = plugVal.asChar(charLen);
	std::string strVal(nameChar, charLen);
	opGraph->name = strVal;
	DEBUGS("SUCCESSFULLY SET NAME")
	// set graph bool node
	bool prevState = data.outputValue(aStGraph).asBool();
	data.outputValue(aStGraph).setBool(!prevState);

	data.setClean(aStGraph);
	data.setClean(aStGraphName);
	data.setClean(plug);
	return s;
}


MStatus StrataGraphNode::legalConnection(const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc,
	bool& isLegal
)		const {
	/* check that only valid things draw from the Strata graph plug - 
	for now we just check that the name is correct.
	If you make a dynamic attribute on a random transform and name it "stGraph",
	then yes, congrats, you can crash maya, very clever
	*/
	if (plug.attribute() != aStGraph) { // only care about graph plug
		return MS::kUnknownParameter;
	}
	MFnAttribute aFn(otherPlug.attribute());
	// if the other plug's name is "stGraph", then we're good
	if (aFn.name() == MString("stGraph")) {
		isLegal = true;
	}
	else { // get denied you criminal bozo
		isLegal = false;
	}
	return MS::kSuccess;
}

/*
Graph item indices are CONSTANT for SINGLE BUILD and EVALUATION of graph.

They are NOT constant across rebuilds, adding or removing nodes.
They also DO NOT correspond to evaluation order, only order they were added to graph.

this is ok.
*/

MStatus StrataGraphNode::connectionMade(const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc
) {
	/* check a connection made to the bool graph plug -
	* reorder and reconnect internal Strata ops if number of nodes
	* in graph changes 
	*/
	DEBUGS("GRAPH connection made")
	if (plug.attribute() != aStGraph) { // only care about the graph plug
		DEBUGS("GRAPH connection made not to graph plug, skipping")
		return MPxNode::connectionMade(plug, otherPlug, asSrc);
	}
	MS s(MS::kSuccess);
	// new connection only appends new op, no need to reorder everything
	/* cast here is actually a bit freaky - 
	MFnDependencyNode.userNode() gets us an MPxNode pointer 
	to the child type object (say StrataAddPointsOpNode) - 
	we want to cast that to StrataOpMixin, 
	which is a separate PARENT class of the child type.
	Hopefully this is fine?
	*/
	StrataOpMixin* mixinPtr;
	s = castToUserNode(otherPlug.node(), mixinPtr);
	if (s != MS::kSuccess) {
		DEBUGS("StrataGraph node could not cast to mixin class pointer on bool plug connected, aborting");
		return s;
	}

	// set references to graph on newly connected node
	mixinPtr->opGraphPtr = opGraph;
	DEBUGS("set graph pointer on mixin");

	// get new op to add to graph
	StrataOp newOp = mixinPtr->createNewOp();
	DEBUGS("got new op from node")
	// add it and get pointer to its spot in graph vector
	mixinPtr->opPtr = opGraph->addOp(newOp);
	DEBUGS("added new op to st graph")
	int newIndex = mixinPtr->opPtr->index;
	DEBUGS("new node index to set: " + std::to_string(newIndex));
	// set the output index plug on the maya node to the index of this new op
	// sure hope this doesn't kick off undue DG evaluation
	MFnDependencyNode depFn(otherPlug.node());
	MPlug indexPlug = depFn.findPlug("stOutput", false, &s);
	indexPlug.setInt(newIndex);

	// update node's internal map of MObjects
	indexMObjHandleMap[newIndex] = MObjectHandle(otherPlug.node());

	return MPxNode::connectionMade(plug, otherPlug, asSrc);
}

MStatus StrataGraphNode::connectionBroken(const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc
) {
	/* if a connection is removed from the graph plug,
	remove the corresponding node's op
	and reorder everything

	*/
	if (plug.attribute() != aStGraph) { // only care about the graph plug
		return MPxNode::connectionBroken(plug, otherPlug, asSrc);
	}
	MS s(MS::kSuccess);

	// get the current index of the disconnected node - should be guaranteed to have one if it's been connected
	MFnDependencyNode depFn(otherPlug.node());
	MPlug indexPlug = depFn.findPlug("stOutput", false, &s);
	int disconnectIndex = indexPlug.asInt();

	// remove all strataOps after the given index
	int nOps = static_cast<int>(opGraph->ops.size());
	opGraph->ops.erase(
		opGraph->ops.begin() + disconnectIndex,
		opGraph->ops.end()
	);

	// go over all nodes, cast them all, get new ops from them, add to graph and set new output indices on maya nodes
	MDGModifier setMayaNodeIndexMod;
	std::vector<MObject*> nodesToShuffle;
	for (int i = disconnectIndex; i < nOps; i++) {
		//
		// cast the outgoing node to its MPxNode, then to its StrataMixin
		StrataOpMixin* mixinPtr;
		MObject nodeMObj = indexMObjHandleMap[i].object();
		MFnDependencyNode depFn(nodeMObj);

		s = castToUserNode(nodeMObj, mixinPtr);
		if (s != MS::kSuccess) {
			DEBUGS("Could not cast plug node " + depFn.name() + " to StrataMixin during graph plug disconnect");
			return MPxNode::connectionBroken(plug, otherPlug, asSrc);
			//continue;
		}
		indexMObjHandleMap.erase(i); // remove this index entry from hash map

		// the actual removed node needs index set to -1, and removed from map
		if (i == disconnectIndex) {
			MPlug outPlug = depFn.findPlug("stOutput", false, &s);
			if (s != MS::kSuccess) {
				DEBUGS("cannot find stOutput plug for REMOVED node");
			}
			setMayaNodeIndexMod.newPlugValueInt(outPlug, -1);
			continue;
		}
		nodesToShuffle.push_back(&nodeMObj);

		// get new op, add to graph vector
		StrataOp* newOp = opGraph->addOp( mixinPtr->createNewOp() );
		// set maya node's index to strata op's index
		// use dgModifier to act all at once
		
		MPlug outPlug = depFn.findPlug("stOutput", false, &s);
		if (s != MS::kSuccess) {
			DEBUGS("cannot find stOutput plug for node");
		}
		setMayaNodeIndexMod.newPlugValueInt(
			outPlug, 
			newOp->index);
		
	}
	// execute modifier, set node indices
	setMayaNodeIndexMod.doIt();
	// unsure if this sets off computation, or if plugs
	// driven by these outputs will have the raw int values propagated
	// for safety we don't rely on it, and crawl DG directly to get values from output plugs

	// at this point we still need to rebuild the index->mayaNode map,
	// and we still need to rebuild Strata input connections based on connections in the Maya graph
	// (individual nodes should already handle their own conversions to strata parametres)

	for (MObject* nodeObj : nodesToShuffle) {
		if ((*nodeObj).isNull()) {
			DEBUGS("could not get node object from saved shuffle pointer");
			continue;
		}

		MFnDependencyNode depFn(*nodeObj);
		int nodeIndex = depFn.findPlug("stOutput", false).asInt();

		// add new MObject handle to index map
		indexMObjHandleMap[nodeIndex] = MObjectHandle(*nodeObj);
		
		// get input array plug
		MPlug strataInputArrPlug = depFn.findPlug("stInput", true, &s);
		if (s != MS::kSuccess) {
			DEBUGS("could not find stInput array plug on node" + depFn.name());
			continue;
		}
		// iterate over child plugs
		for (unsigned int inputLocalIndex = 0;
			inputLocalIndex < strataInputArrPlug.numConnectedElements();
			inputLocalIndex++) {
			MPlug driverPlug = strataInputArrPlug.connectionByPhysicalIndex(inputLocalIndex).source();
			if (driverPlug.isNull()) {
				DEBUGS("found null int driver plug during node shuffle, skipping");
				continue;
			}
			int driverGlobalIndex = driverPlug.asInt();
			
			// finally add this driver index to the strata op node
			opGraph->ops[nodeIndex].inputs.push_back(driverGlobalIndex);
			// it's just that easy
		}
	}
	return MPxNode::connectionBroken(plug, otherPlug, asSrc);
}




MStatus StrataGraphNode::getConnectedStrataOpGraph(
	MObject& thisNodeObj, MObject& graphIncomingConnectionAttr,
	std::weak_ptr<ed::StrataOpGraph>& graphPtr
) {
	/* find the plug for an incoming graph connection, assume that a StrataGraphNode
	is connected, try to get its contained graph.
	If impossible, reset weak pointer
	ABSOLUTELY CHECK MSTATUS here, or face crashes
	*/
	using namespace ed;
	MStatus s(MS::kSuccess);
	MFnDependencyNode thisDepFn(thisNodeObj);
	MPlug incomingConnectionPlug = thisDepFn.findPlug(graphIncomingConnectionAttr, true);
	MObject drivingNodeObj;
	s = getDrivingNode(incomingConnectionPlug, drivingNodeObj);
	if (s == MS::kFailure) {
		//graphPtr = nullptr;
		graphPtr.reset();
		MCHECK(s, "Error getting driving node in getConnectedStrataOpGraph(), RETURNING NULL GRAPH POINTER");
	}
	StrataGraphNode* nodeStructPtr;
	s = castToUserNode<StrataGraphNode>(drivingNodeObj, nodeStructPtr);
	if (s == MS::kFailure) {
		graphPtr.reset();
		MCHECK(s, "Error casting driver node to StrataGraphNode in getConnectedStrataOpGraph(), RETURNING NULL GRAPH POINTER");
	}
	// set maya node's weak pointer to master graph node's shared pointer
	graphPtr = nodeStructPtr->opGraph;
	return s;
}
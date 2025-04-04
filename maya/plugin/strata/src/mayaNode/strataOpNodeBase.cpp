
#include <stdlib.h>
#include <cstdlib>
#include "../macro.h"
#include "../api.h"
#include "strataOpNodeBase.h"
#include "../lib.cpp"


using namespace ed;


// use this to check that we've defined the attribute MObjects once -
// otherwise we just add them to the new nodes
int StrataOpNodeBase::strataAttrsDefined = 0;


//DEFINE_STRATA_STATIC_MOBJECTS(StrataOpNodeBase);

DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataOpNodeBase);


//
//template <typename StrataOpT>
//void StrataOpNodeBase<StrataOpT>::testNoTemplateFn() {}
//
////template<class StrataOpT>
////void StrataOpNodeBase<StrataOpT>::testFn() {}
//
////template<typename NodeT>
////void StrataOpNodeBase::testFn<NodeT>() {}
//
//template<typename StrataOpT, typename NodeT>
//void StrataOpNodeBase<StrataOpT>::testFn() {}
//
//template<typename StrataOpT, typename NodeT>
//void StrataOpNodeBase<StrataOpT>::testFn<NodeT>() {}
//
//template<typename StrataOpT> 
//template<typename NodeT>
//void StrataOpNodeBase<StrataOpT>::testFn<NodeT>() {}
//
//hahahahahahahahahaahaaaaha kill me


//template<typename StrataOpT>
//const std::string StrataOpNodeTemplate<StrataOpT>::getOpNameFromNode(MObject& nodeObj) {

template <typename NodeT>
const int StrataOpNodeBase::getOpIndexFromNode(const MObject& nodeObj) {
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
}
template <typename NodeT>
const std::string StrataOpNodeBase::getOpNameFromNode(MObject& nodeObj) {
	/* return a default name for strata op -
	if nothing defined in string field, use name of node itself*/
	//DEBUGS("GET OP NAME")
	NODELOG("GET OP NAME");
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


const int StrataOpNodeBase::getOpIndexFromNode(const MObject& nodeObj) {
	/* return a default name for strata op -
	if nothing defined in string field, use name of node itself*/
	MFnDependencyNode depFn(nodeObj);
	MStatus s;
	MPlug opNameFieldPlug = depFn.findPlug("stOutput", false, &s);
	if (s.error()) {
		DEBUGS("error getting op index field for node " + depFn.name());
		return -1;
	}
	return opNameFieldPlug.asInt(); // NB - this might cause loops - if so, don't put it in driven.
}
const std::string StrataOpNodeBase::getOpNameFromNode(MObject& nodeObj) {
	/* return a default name for strata op -
	if nothing defined in string field, use name of node itself*/
	MFnDependencyNode depFn(nodeObj);
	MStatus s;
	MPlug opNameFieldPlug = depFn.findPlug("stOpName", false, &s);
	if (s.error()) {
		DEBUGS("error getting op name field for node " + depFn.name());
		return "";
	}
	if (opNameFieldPlug.asString() != "") {
		return opNameFieldPlug.asString().asChar();
	}
	return depFn.name().asChar();
}

MStatus StrataOpNodeBase::setFreshGraph(MObject& nodeObj) {
	MS s;
	DEBUGS("base set fresh graph")
	opGraphPtr = std::make_shared<StrataOpGraph>();
	// need to extend in templated class to add a new node to the graph
	return s;
}

void StrataOpNodeBase::postConstructor(MObject& nodeObj) {
	/* ensure graph pointer is reset*/
	DEBUGS("Base postConstructor")
	addedToGraph = true;
	//return;
	MS s = setFreshGraph(nodeObj);
}

MStatus StrataOpNodeBase::syncIncomingGraphConnections(MObject& nodeObj) {
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
	MPlug graphInPlug = depFn.findPlug(aStGraph, true, &s);
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
	//// if no 0 entry found, create a new graph
	//if (incomingGraphPtrs.find(0) == incomingGraphPtrs.end()) {
	//	setFreshGraph();
	//}
	return s;
}

MStatus StrataOpNodeBase::syncIncomingGraphData(MObject& nodeObj) {
	/* copy incoming graph data - extend to also add new node into graph
	* check through incoming map - if 0 entry found, copy it to a new object
	*/
	MS s = MS::kSuccess;
	if (incomingGraphPtrs.find(0) == incomingGraphPtrs.end()) {
		// 0 not found, make new graph
		s = setFreshGraph(nodeObj);
		return s;
	}
	if (incomingGraphPtrs[0].expired()) {
		// 0 weak pointer expired (somehow), make new graph
		setFreshGraph(nodeObj);
		return s;
	}
	opGraphPtr = incomingGraphPtrs[0].lock().get()->cloneShared<StrataOpGraph>();
	return s;
}

MStatus StrataOpNodeBase::compute(
	MObject& nodeObj, 
	const MPlug& plug, 
	MDataBlock& data) {
	MS s(MS::kSuccess);
	// sync op name
	if (plug.attribute() == aStOpNameOut) {
		syncOpNameOut(nodeObj, data);
		data.setClean(plug);
		return s;
	}

	// copy graph data if it's dirty
	if (!data.isClean(aStGraph)) {
		s = syncIncomingGraphData(nodeObj);

	}
	return s;
}


//template<typename StrataOpT>
//MStatus StrataOpNodeTemplate<StrataOpT>::legalConnection(
MStatus StrataOpNodeBase::legalConnection(
	//MObject& nodeObj,
	const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc,
	bool& isLegal
) {
	// check that an input to the array only comes from another strata maya node - 
	// can't just connect a random integer
	DEBUGS("Base legalConnection")
	if (MFnAttribute(otherPlug.attribute()).name() == "stOutput") {
		if (otherPlug.attribute() != aStOutput) {
			DEBUGS("WARNING - attribute MOBJECTS are not equivalent or multiply redefined, fix")
		}
	}

	if (plug.attribute() == aStGraph) { // main graph input
		if (otherPlug.node() == plug.node()) { // no feedback loops
			isLegal = false;
			return MS::kSuccess;
		}
		if (MFnAttribute(otherPlug.attribute()).name() == "stOutput") {
			isLegal = true;
			return MS::kSuccess;
		}
	}

	if (MFnAttribute(plug.attribute()).name() == "stInput") {
		if (otherPlug.node() == plug.node()) { // no feedback loops
			isLegal = false;
			return MS::kSuccess;
		}
		if (MFnAttribute(otherPlug.attribute()).name() == "stOutput") {
			isLegal = true;
			return MS::kSuccess;
		}
	}
	return MStatus::kUnknownParameter;
}

//template<typename StrataOpT>
//MStatus StrataOpNodeTemplate<StrataOpT>::connectionMade(const MPlug& plug,
MStatus StrataOpNodeBase::connectionMade(
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
	s = syncIncomingGraphConnections(plug.node());
	MCHECK(s, "ERROR refreshing graph pointer from incoming node connection");
	return s;
}

//template<typename StrataOpT>
//MStatus StrataOpNodeTemplate<StrataOpT>::connectionBroken(const MPlug& plug,
MStatus StrataOpNodeBase::connectionBroken(
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
	syncIncomingGraphConnections(plug.node());
	return s;
}



//template<typename StrataOpT>
//void StrataOpNodeTemplate<StrataOpT>::postConstructor(MObject& nodeObj) {
//	/* ensure graph pointer is reset*/
//	DEBUGS("Template postConstructor");
//
//	StrataOpNodeBase::postConstructor(nodeObj);
//}

template<typename StrataOpT>
MStatus StrataOpNodeTemplate<StrataOpT>::syncIncomingGraphData(MObject& nodeObj) {
	/* copy graph, and add new node to it*/
	MStatus s = StrataOpNodeBase::syncIncomingGraphData(nodeObj);
	MCHECK(s, "Error copying graph data from OpNodeBase, halting");
	StrataOpT* opPtr = createNewOp(nodeObj, s);
	MCHECK(s, "Error adding new op to new graph");
	
}

template<typename StrataOpT>
MStatus StrataOpNodeTemplate<StrataOpT>::syncOpInputs(StrataOpT* op, const MObject& node) {
	// check through input plugs on maya node, 
	// triggered when input connections change
	MStatus s(MS::kSuccess);


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
	opGraphPtr->nodeInputsChanged(op->index);

	return s;
}





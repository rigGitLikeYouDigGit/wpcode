

#include "../macro.h"
#include "../api.h"
#include "strataOpNodeBase.h"
#include "../lib.cpp"


using namespace ed;

//DEFINE_STRATA_STATIC_MOBJECTS(StrataOpNodeBase);

DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataOpNodeBase);

MStatus StrataOpNodeBase::addStrataAttrs(
	std::vector<MObject>& driversVec,
	std::vector<MObject>& drivenVec
) {
	MS s(MS::kSuccess);

	MFnNumericAttribute nFn;
	MFnCompoundAttribute cFn;
	MFnTypedAttribute tFn;

	// all nodes connected directly to master graph node
	//aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kBoolean);
	//nFn.setReadable(false);
	//nFn.setStorable(false);
	//nFn.setChannelBox(false);
	//nFn.setKeyable(false);

	aStInput = nFn.create("stInput", "stInput", MFnNumericData::kInt);
	nFn.setReadable(false);
	nFn.setArray(true);
	nFn.setUsesArrayDataBuilder(true);
	nFn.setIndexMatters(true);
	nFn.setDefault(-1);
	// strata inputs use physical order always - there's no reason ever to have an empty entry here
	nFn.setDisconnectBehavior(MFnAttribute::kDelete);

	aStOpName = tFn.create("stOpName", "stOpName", MFnData::kString);
	tFn.setDefault(MFnStringData().create(""));
	

	//T::aStInputAlias = tFn.create("stInputAlias", "stInputAlias", MFnData::kString);
	//tFn.setReadable(false);
	//tFn.setArray(true);
	//tFn.setUsesArrayDataBuilder(true);
	//tFn.setIndexMatters(true);

	// output index of st op node, use as output to all 
	// -1 as default so it's obvious when a node hasn't been initialised, connected to graph, etc
	aStOutput = nFn.create("stOutput", "stOutput", MFnNumericData::kInt, -1);
	nFn.setReadable(true);
	nFn.setWritable(false);

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

	std::vector<MObject> drivers = {
		//aStGraph,
		aStInput,
		aStOpName
		//aStParam,
		//aStElData
	};
	driversVec.insert(driversVec.end(), drivers.begin(), drivers.end());

	std::vector<MObject> driven = {
		aStOutput
	};
	drivenVec.insert(drivenVec.end(), driven.begin(), driven.end());

	return s;
}



MStatus StrataOpNodeBase::syncOpInputs(ed::StrataOp* op, const MObject& node) {
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

MStatus StrataOpNodeBase::compute(const MPlug& plug, MDataBlock& data) {
	/* base computation for each op nodes
	consider common stages, like pulling data from node plugs, syncing 
	strata ops
	
	*/
	MS s(MS::kSuccess);

	// check if plug is already computed
	if (data.isClean(plug)) {
		return s;
	}
	//// check if graph connection has been lost
	//if (opGraphPtr.expired()) {
	//	data.setClean(plug);
	//	return s;
	//}

	// check if input data/indices have changed
	if (!data.isClean(aStInput)) {

	}
	return s;

}

const std::string StrataOpNodeBase::getOpNameFromNode(MObject& nodeObj) {
	/* return a default name for strata op -
	if nothing defined in string field, use name of node itself*/
	MFnDependencyNode depFn(nodeObj);
	MStatus s;
	MPlug opNameFieldPlug = depFn.findPlug(aStOpName, false, &s);
	if (s.error()) {
		DEBUGS("error getting op name field for node " + depFn.name());
		return "";
	}
	if (opNameFieldPlug.asString() != "") {
		return opNameFieldPlug.asString().asChar();
	}
	return depFn.name().asChar();
}

StrataOp* StrataOpNodeBase::createNewNode(MObject& mayaNodeMObject) {
	return opGraphPtr->addNode<StrataOpT>(
		getOpNameFromNode(mayaNodeMObject)
	);
}

MStatus StrataOpNodeBase::refreshGraphPtr(StrataOpGraph* otherGraph) {
	/* if other graph is not a nullptr,
	copy graph, re-add this node's op, make all connections*/
	MS s;
	if (otherGraph == nullptr) { // new graph, empty op
		opGraphPtr = std::make_unique<StrataOpGraph>();

		

	}
	return s;
}

void StrataOpNodeBase::postConstructor() {
	/* ensure graph pointer is reset*/
	opGraphPtr = std::make_unique<ed::StrataOpGraph>();
}

MStatus StrataOpNodeBase::legalConnection(
	const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc,
	bool& isLegal
) {
	// check that an input to the array only comes from another strata maya node - 
	// can't just connect a random integer
	if (plug.attribute() == aStInput) {
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


MStatus StrataOpNodeBase::connectionMade(const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc
) {
	/* on connection made from another strata op node, 
	copy the */
	MStatus s = MS::kSuccess;
	if (plug.attribute() != aStInput) {
		return s;
	}
	
	StrataOpNodeBase* otherNodePtr;
	s = castToUserNode<StrataOpNodeBase>(otherPlug.node(), otherNodePtr);
	MCHECK(s, "ERROR casting incoming strata connection to StrataOpNodeBase");
	s = refreshGraphPtr(otherNodePtr->opGraphPtr.get());
	MCHECK(s, "ERROR refreshing graph pointer from incoming node connection");
	return s;
}

MStatus StrataOpNodeBase::connectionBroken(const MPlug& plug,
	const MPlug& otherPlug,
	bool 	asSrc
) {
	MStatus s = MS::kSuccess;
	if (plug.attribute() != aStInput) {
		return s;
	}
	refreshGraphPtr(nullptr);
	return s;
}


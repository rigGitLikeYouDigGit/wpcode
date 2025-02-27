

#include "../macro.h"
#include "../api.h"
#include "strataOpNodeBase.h"
#include "../lib.cpp"


using namespace ed;



//MString StrataGraphNode::kNODE_NAME;
//MString StrataOpNodeBase::kNODE_NAME = MString("strataOpNodeBase");
//MTypeId StrataOpNodeBase::kNODE_ID = MTypeId(0x00122CA1);

MObject StrataOpNodeBase::aStGraph; // opgraph connection
MObject StrataOpNodeBase::aStParent; // int index for parent element

MObject StrataOpNodeBase::aStInput; // array of int node ids, first is always main
MObject StrataOpNodeBase::aStInputAlias; // string array to use for inputs - indexMatters

MObject StrataOpNodeBase::aStOpIndex; // index of this op in strata
MObject StrataOpNodeBase::aStOutput; // out strata manifold, bool plug
MObject StrataOpNodeBase::aStManifoldData; // use to evaluate manifold elements at this point in graph

MObject StrataOpNodeBase::aStParam; // custom params for node
MObject StrataOpNodeBase::aStParamExpression; // expression string attribute for each one


//MStatus StrataOpNodeBase::initialize() {
template<typename T>
MStatus StrataOpNodeBase::addStrataAttrs(){
	MS s(MS::kSuccess);

	MFnNumericAttribute nFn;
	MFnCompoundAttribute cFn;
	MFnTypedAttribute tFn;

	// all nodes connected directly to master graph node
	aStGraph = nFn.create("stGraph", "stGraph", MFnNumericData::kBoolean);
	nFn.setReadable(false);

	aStInput = nFn.create("stInput", "stInput", MFnNumericData::kInt);
	nFn.setReadable(false);
	nFn.setArray(true);
	nFn.setUsesArrayDataBuilder(true);
	nFn.setIndexMatters(true);
	nFn.setDefault(-1);

	aStInputAlias = tFn.create("stInputAlias", "stInputAlias", MFnData::kString);
	tFn.setReadable(false);
	tFn.setArray(true);
	tFn.setUsesArrayDataBuilder(true);
	tFn.setIndexMatters(true);

	aStOutput = nFn.create("stOutput", "stOutput", MFnNumericData::kInt);
	nFn.setReadable(true);
	nFn.setWritable(false);

	// add attributes
	std::vector<MObject> drivers = {
		aStGraph, aStInput
	};

	std::vector<MObject> driven = {
		aStOutput
	};

	addAttributes<T>(drivers);
	addAttributes<T>(driven);
	setAttributesAffect<T>(drivers, driven);

	return s;
}



#pragma once

#include <cstdint>


#include <maya/MPxNode.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MEventMessage.h>
#include <maya/MGlobal.h>


#include "../stratacore/manifold.h"
#include "../stratacore/opgraph.h"


using namespace ed;

class StrataGraphNode : public MPxNode {
	/* master maya node representing single strata
	op graph - output its pointer address in memory as long
	NOPE - turns out a pointer in c++ won't fit in a long, and has to be
	a special uintptr_t integer type

	can't find any resources on splitting it into 2 integers, so to connect later
	nodes, to this one, we'll use the normal bool balancewheel method
	*/


public:
	StrataGraphNode() {}
	virtual ~StrataGraphNode() {}



	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	// single opgraph for this object
	// statewise, each individual op node should first check that
	StrataOpGraph opGraph;

	// attribute MObjects
	static MObject aStGraphName;
	static MObject aStGraph;


	static void* creator() {
		StrataGraphNode* newObj = new StrataGraphNode();
		return newObj;
	}

	static MStatus initialize() {
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

		addAttribute(aStGraphName);
		addAttribute(aStGraph);

		attributeAffects(aStGraphName, aStGraph);

		return s;
	}

	virtual void postConstructor() {
		/* initialise new strata op graph with this node's name*/
		opGraph = StrataOpGraph();
	}

	virtual MStatus compute(const MPlug& plug, MDataBlock& data) {
		// set the op graph's name, then set its memory address
		// first check if plug is clean
		MS s(MS::kSuccess);
		if (data.isClean(plug)) {
			return s;
		}

		// set name
		opGraph.name = std::string(data.inputValue(aStGraphName).asString().asChar());

		// set graph bool node
		data.outputValue(aStGraph).setBool(!data.outputValue(aStGraph).asBool());

		data.setClean(aStGraph);
		data.setClean(aStGraphName);
		data.setClean(plug);
		return s;
	}


};

MObject StrataGraphNode::aStGraph;
MObject StrataGraphNode::aStGraphName;

//MString StrataGraphNode::kNODE_NAME;
MString StrataGraphNode::kNODE_NAME = MString("strataGraph");
MTypeId StrataGraphNode::kNODE_ID = MTypeId(0x00122CA0);



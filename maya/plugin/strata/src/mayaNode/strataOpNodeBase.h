#pragma once

#include <maya/MObject.h>
#include "../MInclude.h"


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


//class StrataOpNodeBase : MPxNode {
class StrataOpNodeBase {

public:

	/*StrataOpNodeBase() {}
	virtual ~StrataOpNodeBase() {}

	static void* creator() {
		StrataOpNodeBase* newObj = new StrataOpNodeBase();
		return newObj;
	}

	bool isAbstractClass() const {
		return true;
	}

	static MStatus initialize();*/

	template<typename T>
	static MStatus addStrataAttrs();

	//static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	//static MString kNODE_NAME;// = MString("curveFrame");

	static MObject aStGraph; // opgraph connection
	static MObject aStParent; // int index for parent element

	static MObject aStInput; // array of int node ids, first is always main
	static MObject aStInputAlias; // string array to use for inputs - indexMatters

	static MObject aStOpIndex; // index of this op in strata
	static MObject aStOutput; // out strata manifold, bool plug
	static MObject aStManifoldData; // use to evaluate manifold elements at this point in graph

	static MObject aStParam; // custom params for node
	static MObject aStParamExpression; // expression string attribute for each one

};





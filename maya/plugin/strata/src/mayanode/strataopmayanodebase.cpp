#pragma once

#include <maya/MObject.h>


/*
Base class for all strata operation nodes - for now we try to mirror the maya graph 1:1 
in structure, if not always in evaluation

- one master plug for the graph flow

keep evaluation in step for now too, otherwise Undo gets insane
*/


class StrataOpNodeBase {

public:
	static MObject aStGraph;
	static MObject aStParent;
	


};


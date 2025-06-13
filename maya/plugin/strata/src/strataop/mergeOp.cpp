
#include "mergeOp.h"
#include "../stringLib.h"
#include "../lib.h"
#include "../libEigen.h"

using namespace ed;
using namespace ed::expns;

Status StrataMergeOp::makeParams() {
	Status s;
	return s;
}


Status StrataMergeOp::eval(StrataManifold& value,
	EvalAuxData* auxData, Status& s)
{
	/*
	*/
	DEBUGSL("MERGE OP EVAL");
	auto graph = getGraphPtr();

	//StrataOpGraph* graphPtr = static_cast<StrataOpGraph*>( graphPtr) ;
	for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
		if (inputs[i] == -1) { continue; }
		value.mergeOther(graph->results[inputs[i]], StrataManifold::MERGE_OVERWRITE, s);
		CWRSTAT(s, "Error merging strata manifold " + std::to_string(i) + " on op " + name);
	}
	
	// transform if needed
	if (!worldMat.isApprox(Affine3f::Identity())) {
		value.transform(worldMat);
	}

	return s;
}

//Status evalTopo(StrataManifold& manifold, Status& s) {
//	return s;
//}
//
//Status evalData(StrataManifold& manifold, Status& s) {
//	return s;
//}
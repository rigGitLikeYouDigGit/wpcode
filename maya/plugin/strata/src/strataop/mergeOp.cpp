
#include "mergeOp.h"
#include "../stringLib.h"
#include "../lib.h"
#include "../libEigen.h"
#include "../logger.h"

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
	LOG("MERGE OP EVAL");
	auto graph = getGraphPtr();

	//StrataOpGraph* graphPtr = static_cast<StrataOpGraph*>( graphPtr) ;
	for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
		if (inputs[i] == -1) { continue; }
		StrataOp* prevOp = graph->getNode<StrataOp>(inputs[i]);

		l("before merge input from op: " + str(inputs[i]) + " " + prevOp->name);
		value.mergeOther(graph->results[inputs[i]], StrataManifold::MERGE_OVERWRITE, s);
		CWRSTAT(s, "Error merging strata manifold " + std::to_string(i) + " on op " + name);
	}
	
	// transform if needed
	if (!worldMat.isApprox(Affine3f::Identity())) {
		value.transform(worldMat);
	}

	return s;
}

StrataMergeOp* StrataMergeOp::clone_impl() const { 
	LOG("MERGE OP CLONE");
	return StrataOp::clone_impl<StrataMergeOp>(); };

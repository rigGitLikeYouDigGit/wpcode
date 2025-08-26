
#include "expGraph.h"
using namespace strata;
using namespace expns;


Status ExpOpNode::eval(std::vector<ExpValue>& value, EvalAuxData* auxData, Status& s) {
	/* pull in ExpValues from input nodes, join all arguments together -
	MAYBE there's a case for naming blocks of arguments but that gets insane -
	python kwargs seem a bit excessive for now

	pass into atom as arguments*/
	std::vector<ExpValue> arguments;
	ExpAuxData* expAuxData = static_cast<ExpAuxData*>(auxData);

	if (graphPtr == nullptr) {
		STAT_ERROR(s, "UNABLE TO CAST NODE GRAPHPTR TYPE TO EXPGRAPH*");
	}
	//ExpGraph* testGraphPtr = static_cast<ExpGraph*>((node->graphPtr));
	//ExpGraph* testGraphPtr = reinterpret_cast<ExpGraph*>((node->graphPtr));
	/*if (testGraphPtr == nullptr) {
		STAT_ERROR(s, "UNABLE TO CAST NODE GRAPHPTR TYPE TO EXPGRAPH*");
	}*/

	ExpGraph* graphPtr = getGraphPtr();
	for (int index : inputs) {
		arguments.insert(arguments.end(),
			graphPtr->results[index].begin(),
			graphPtr->results[index].end());
	}

	Status result = expAtomPtr->eval(
		arguments,
		//auxData,
		expAuxData,
		value,
		s);
	return s;
}



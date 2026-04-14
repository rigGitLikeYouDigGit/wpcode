

#include "resultAtom.h"
#include "expValue.h"
#include "expGraph.h"

using namespace strata;
using namespace expns;

Status ResultAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
{
	if (argList.size() == 0) { // nothing to do
		return s;
	}
	// check that all incoming arguments have the same type
	std::string firstType = argList[0].t;
	//for (auto& arg : argList) {
	for (int i = 1; i < argList.size(); i++) {
		ExpValue arg = argList[i];
		if (arg.t != firstType) {
			STAT_ERROR(s, "Mismatch in value types for result, halting");
		}
		argList[0].extend({ arg });
	}

	//result.push_back(argList[0]);
	return s;
}

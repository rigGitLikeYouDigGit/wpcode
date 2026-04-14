
#include "plusAtom.h"

#include "expParse.h"

#include "expGraph.h"
#include "expValue.h"

using namespace strata;
using namespace expns;


Status PlusAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
{
	if (!(argList.size() == 2)) { // check only name of variable and variable value passed
		STAT_ERROR(s, "Can only add 2 values together ");
	}

	ExpValue v;

	v.copyOther(argList[1]);

	// create variable in expression status / scope
	auxData->expStatus->varMap[v.varName] = v;

	return s;
}
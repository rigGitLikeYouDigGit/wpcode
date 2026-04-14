
#include "expAtom.h"
#include "expValue.h"

using namespace strata;
using namespace expns;

Status ExpAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
	std::vector<ExpValue>& result,
	Status& s)
{
	LOG("EXPATOM base eval - probably wrong");
	result = argList;
	return s;
}



// function to insert this op in graph
Status ExpAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int& outNodeIndex,
	Status& s
) {
	LOG("ExpAtom parse");
	STAT_ERROR(s, "BASE EXP ATOM PARSE called");
}

// function to insert this op in graph
Status PrefixParselet::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int& outNodeIndex,
	Status& s
) {
	LOG("PrefixParselet parse");
	STAT_ERROR(s, "BASE PREFIX ATOM PARSE called");
}

Status InfixParselet::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	LOG("InfixParselet parse: " + str(token) + " " + str(srcString) + " " + str(leftIndex) + str(outNodeIndex));
	STAT_ERROR(s, "BASE INFIX ATOM PARSE called");
}





#include "expParse.h"
#include "expElCompare.h"

#include "../stratacore/manifold.h" // fine to tightly couple here, this expression language is never meant to be standalone
#include "../stratacore/libManifold.h"

#include "../logger.h"

using namespace strata;
using namespace strata::expns;


Status GreaterThanAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	/* probably no need for custom handling here, could have a common
	function for all infixes
	*/
	DirtyNode* newNode = nullptr;
	DirtyNode* leftNode = graph.getNode(leftIndex);
	if (leftNode == nullptr) {
		STAT_ERROR(s, "No left side for operation " + this->srcString);
	}

	newNode = graph.addNode<GreaterThanAtom>();
	outNodeIndex = newNode->index;

	newNode->inputs.push_back(leftIndex);

	int rightIndex = -1;
	s = parser.parseExpression(graph, rightIndex);
	if (rightIndex == -1) {
		STAT_ERROR(s, "No right side for operation " + this->srcString);
	}

	// add name of function to call inputs
	newNode->inputs.push_back(rightIndex);

	return s;
}

Status GreaterThanAtom::eval(
	std::vector<ExpValue>& argList,
	ExpAuxData* auxData,
	std::vector<ExpValue>& result, Status& s
) {
	/* for each element in left, check for intersection with each in right
	if none found, just return that element

	TODO: how do we differentiate this from just a normal greaterThan bool check

	TODO: should we allow looser operations only considering nearest-point?

	*/
	ExpValue& left = argList[0];
	ExpValue& right = argList[1];

	std::vector<int> leftVals;
	std::vector<int> rightVals;
	std::vector<int> outVals;
	s = auxData->expValuesToElements(s, left, leftVals);
	s = auxData->expValuesToElements(s, right, rightVals);

	s = elementGreaterThan(s, *auxData->manifold,
		leftVals, rightVals, outVals);
	ExpValue resultVal;

	/* should exp ops prefer returning name lists or numbers? or both?*/
	std::vector<std::string> nameList(outVals.size());
	for (int i = 0; i < static_cast<int>(outVals.size()); i++) {
		nameList[i] = auxData->manifold->getEl(i)->name;
	}
	resultVal.stringVals = nameList;
	result.push_back(resultVal);
	return s;
}

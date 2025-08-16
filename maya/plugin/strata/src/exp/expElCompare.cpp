



#include "expParse.h"
#include "expElCompare.h"

#include "../stratacore/manifold.h" // fine to tightly couple here, this expression language is never meant to be standalone

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

	
	return s;
}

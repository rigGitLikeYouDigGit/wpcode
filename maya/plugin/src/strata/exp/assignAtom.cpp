
#include "assignAtom.h"

#include "expParse.h"

#include "expGraph.h"
#include "expValue.h"

#include "nameAtom.h"

using namespace strata;
using namespace expns;



Status AssignAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	int right;
	s = parser.parseExpression(graph, right, Precedence::ASSGIGNMENT - 1);
	CWRSTAT(s, "error parsing right side of assignment: " + token.lexeme());
	DirtyNode* newNode = graph.addNode<AssignAtom>();

	// left index should be a name, and right index should be the value
	ExpOpNode* leftNode = static_cast<ExpOpNode*>(graph.getNode(leftIndex));
	NameAtom* leftOp = dynamic_cast<NameAtom*>(leftNode->expAtomPtr.get());
	if (!leftOp) {
		STAT_ERROR(s, "error reinterpreting left op in assignment, could not recover NameAtom from input node");
	}
	newNode->inputs.push_back(leftIndex);
	newNode->inputs.push_back(right);

	// set the index of this variable in the parse state, as this node is most recent
	// to modify it
	/* feels a bit weird to reach this far back up to get the exp object
	*/
	graph.exp->parseStatus.varIndexMap[leftOp->strName] = newNode->index;
	outNodeIndex = newNode->index;

	return s;
}

Status AssignAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
{
	/*TODO: could allow multiple values here for concatenation -
	in that case we would only need to check that types match between them

	can we use this for keyword arguments?
	*/
	if (!(argList.size() == 2)) { // check only name of variable and variable value passed
		STAT_ERROR(s, "Can only assign single ExpValue to variable, not 0 or multiple");
	}
	//argList[0].varName = argList[1].strVal;
	ExpValue v;
	v.varName = argList[0].stringVals[0];
	v.copyOther(argList[1]);

	// create variable in expression status / scope
	auxData->expStatus->varMap[v.varName] = v;

	// copy left-hand into this node's result, as the value of this variable at this moment
	result.insert(result.begin(), argList.begin() + 1, argList.end());
	return s;
}

#include "callAtom.h"
#include "expGraph.h"
#include "expParse.h"
#include "expValue.h"

using namespace strata;
using namespace expns;


Status CallAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	// check if name of function is the graph's result node - 
	// add each separate argument as expression results
	LOG("callAtom parse");
	DirtyNode* newNode = nullptr;
	DirtyNode* leftNode = graph.getNode(leftIndex);

	newNode = graph.addNode<CallAtom>();
	outNodeIndex = newNode->index;

	// add name of function to call inputs
	newNode->inputs.push_back(leftIndex);

	// parse arg lists and add to input list
	if (!parser.match(Token::Kind::RightParen)) {
		do {
			int argIndex = -1;
			s = parser.parseExpression(graph, argIndex);
			CWRSTAT(s, "error parsing arg from CallAtom, halting");
			newNode->inputs.push_back(argIndex);
		} while (parser.match(Token::Kind::Comma));
		parser.consume(Token::Kind::RightParen, s);
		CWRSTAT(s, "error finding rightParen for callAtom");
	}
	return s;
}



Status CallAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
{
	if (!(argList.size() == 2)) { // check only name of variable and variable value passed
		STAT_ERROR(s, "Can only assign single ExpValue to variable, not 0 or multiple");
	}
	//argList[0].varName = argList[1].strVal;
	ExpValue v;
	v.varName = argList[0].stringVals[0];
	v.copyOther(argList[1]);

	// create variable in expression status / scope
	auxData->expStatus->varMap[v.varName] = v;

	return s;
}




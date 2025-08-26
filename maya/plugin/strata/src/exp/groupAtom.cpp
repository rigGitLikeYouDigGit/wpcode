
#include "groupAtom.h"
#include "expValue.h"
#include "expGraph.h"
#include "expParse.h"

using namespace strata;
using namespace expns;

Status GroupAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
{
	/* merge incoming values into one -
	* result will already have entry from input 0
	*/
	for (auto i = 1; i < argList.size(); i++) {
		result.push_back(argList[i]);
	}
	return s;

}



Status GroupAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	//int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	/* copying logic from CallAtom parse -
	* a Group atom just adds single value to graph, unless multiple specified -
	* emulate tuple in Python
	*/

	LOG("groupAtom parse");

	DirtyNode* newNode = graph.addNode<GroupAtom>();
	outNodeIndex = newNode->index;


	// parse arg lists and add to input list
	if (!parser.match(Token::Kind::RightParen)) {
		do {
			int argIndex = -1;
			s = parser.parseExpression(graph, argIndex, 0);
			CWRSTAT(s, "error parsing arg from CallAtom, halting");
			newNode->inputs.push_back(argIndex);
		} while (parser.match(Token::Kind::Comma) || parser.match(Token::Kind::Space));
		parser.consume(Token::Kind::RightParen, s);
		CWRSTAT(s, "error finding rightParen for callAtom");
	}
	return s;

	//s = parser.parseExpression(graph, outNodeIndex, 0);
	//parser.consume(Token::Kind::RightParen, s);
	//return s;
}

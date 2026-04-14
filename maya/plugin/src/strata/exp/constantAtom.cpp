

#include "expValue.h"
#include "constantAtom.h"
#include "expParse.h"
#include "expGraph.h"

using namespace strata;
using namespace expns;


Status ConstantAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int& outNodeIndex,
	Status& s
) {

	ExpOpNode* newNode = graph.addNode<ConstantAtom>();
	outNodeIndex = newNode->index;
	ConstantAtom* op = static_cast<ConstantAtom*>(newNode->expAtomPtr.get());
	switch (token.getKind()) {
	case Token::Kind::String: {
		op->literalStr = token.lexeme();
	}
	case Token::Kind::Number: {
		op->literalVal = std::stof(token.lexeme());
	}
	default: {
		STAT_ERROR(s, "Unknown token kind to Constant Atom, halting");
	}
	}
	return s;
}

Status ConstantAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
{
	ExpValue v;
	if (!literalStr.empty()) {
		v.stringVals = { literalStr };
	}
	else { v.numberVals = { literalVal }; }
	//v.dims = { 1 };
	result.push_back(v);
	return s;
}


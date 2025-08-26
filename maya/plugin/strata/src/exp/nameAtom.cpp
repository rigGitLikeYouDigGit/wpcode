

#include "nameAtom.h"
#include "expValue.h"
#include "expGraph.h"

using namespace strata;
using namespace expns;


Status NameAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int& outNodeIndex,
	Status& s
) {
	/* lookup first to reuse nodes if possible */
	//LOG("NAME parse: " + str(token) + " " + str(outNodeIndex));
	ExpOpNode* newNode = graph.getNode<ExpOpNode>("NAME_" + token.lexeme());
	if (newNode == nullptr) {
		newNode = graph.addNode<NameAtom>("NAME_" + token.lexeme());

		NameAtom* op = static_cast<NameAtom*>(newNode->expAtomPtr.get());
		op->strName = token.lexeme();
	}
	outNodeIndex = newNode->index;

	return s;
}

// depending on use, this name will either be set or retrieved by the next operation in graph
Status NameAtom::eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
	std::vector<ExpValue>& result,
	Status& s)
{
	ExpValue v;
	v.stringVals = { strName };
	v.varName = strName;
	result.push_back(v);
	return s;
}


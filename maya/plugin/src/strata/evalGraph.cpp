#include "evalGraph.h"

using namespace strata;

struct TestLogic : EvalLogic {
	static constexpr const char* typeName = "test";
	Status& eval(void* nodePtr, void* valuePtr, void* auxData, Status& s) {
		LOG("TEST LOGIC EVAL");
		int* value = static_cast<int*>(valuePtr);
		*value = 42;
		return s;
	}
};

// Explicit template instantiations
template struct EvalGraphBase<int, EvalNode<int, std::variant<TestLogic>>, EvalGraphTest>;
template struct EvalNode<int, std::variant<TestLogic>>;
static EvalGraphTest testGraph;
static auto* n = testGraph.addNode("a", TestLogic{});	
static int testid = n->index;
static EvalNode<int, std::variant<EvalLogic>>* getN = testGraph.getNode(testid);
void evalGraph_dummy() {}
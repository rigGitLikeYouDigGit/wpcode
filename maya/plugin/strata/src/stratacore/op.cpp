

#include "op.h"
#include "opgraph.h"

using namespace ed;

StrataOp* StrataOp::getOp(int opIndex) {
	return &(graph->ops[opIndex]);
}
StrataOp* StrataOp::getOp(std::string opName) {
	return &(graph->ops[graph->opNameIndexMap[opName]]);
}

//ExpGrammar* StrataOp::expGrammar(&baseGrammar);
//StrataOp::expGrammar = &baseGrammar;

void StrataOp::signalIOChanged() {
	/*reach back out to graph, set its structure dirty flag.
	* yes I KNOW this is crazy dependency, maybe the graph should be the only thing that
	* tracks IO on nodes?
	* maybe we should do a proper signalling system?
	*/
	graph->graphChanged = true;

}
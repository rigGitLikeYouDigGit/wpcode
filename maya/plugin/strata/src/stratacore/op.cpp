

#include "op.h"
#include "opgraph.h"

using namespace ed;
//
//StrataOp* StrataOp::getOp(int opIndex) {
//	return &(graph->ops[opIndex]);
//}
//StrataOp* StrataOp::getOp(std::string opName) {
//	return &(graph->ops[graph->opNameIndexMap[opName]]);
//}
//
////ExpGrammar* StrataOp::expGrammar(&baseGrammar);
////StrataOp::expGrammar = &baseGrammar;
//
//void StrataOp::signalIOChanged() {
//	/*reach back out to graph, set its structure dirty flag.
//	* yes I KNOW this is crazy dependency, maybe the graph should be the only thing that
//	* tracks IO on nodes?
//	* maybe we should do a proper signalling system?
//	*/
//	graph->graphChanged = true;
//
//}

Status& StrataOp::runBackPropagation(
	Status& s,
	StrataOp* fromNode,
	StrataManifold& finalManifold,
	SAtomBackDeltaGroup deltaGrp,
	StrataAuxData& auxData
) {
	/* overall top-level back prop function
	assume deltas have already been gathered outside of this

	from that work out what nodes created the affected elements.
	check nodes breadth-first from output node backwards,
	calling prop methods on those nodes if they created elements in delta group
	*/
	//std::set<StrataOp*> toVisit({ fromNode });
	//std::set<StrataOp*> nextToVisit; // don't know how to do breadth-first in leet code

	// get generations in history of nodes
	LOG("OP BACK-PROPAGATION: " + fromNode->name);
	StrataOpGraph* graph = getGraphPtr();
	std::vector<std::vector<int>> generations = graph->nodesInHistory(fromNode->index, true);

	/* backwards pass, getting target deltas for elements/ nodes to match
	*/
	l("back prop backwards pass");
	for (int i = 0; i < static_cast<int>(generations.size()); i++) {

		std::vector<int>& toVisit = generations[i];
		std::vector<SAtomBackDeltaGroup> resultDeltas(toVisit.size()); // results of this iteration of back-prop
		for (int n = 0; n < static_cast<int>(toVisit.size()); n++) { // parallel this
			StrataOp* op = graph->getNode<StrataOp>(toVisit[n]);
			resultDeltas[n] = op->bestFitBackDeltas(&s, finalManifold, deltaGrp);
			CRMSG(s, "error running back propagation on node " + op->name);
		}

		// collate delta fronts
		for (int n = 1; n < static_cast<int>(resultDeltas.size()); n++) {
			resultDeltas[0].mergeOther(resultDeltas[n]);
		}
		deltaGrp = resultDeltas[0]; // need to do a copy here because we create the intermediate vals in the loop scope
	}

	/* now forwards pass, eval'ing nodes and setting final offsets
	*/
	l("back prop forwards pass");
	for (int i = 0; i < static_cast<int>(generations.size()); i++) {

		std::vector<int>& toVisit = *(generations.rbegin() + i);
		for (int n = 0; n < static_cast<int>(toVisit.size()); n++) { // parallel this
			StrataOp* op = graph->getNode<StrataOp>(toVisit[n]);
			l("eval op for forwards prop: " + op->name);
			s = graph->evalGraph(s, toVisit[n], &auxData); // eval node with new best-fit params
			//StrataManifold& m = op->value();
			l("before op back offsets");
			s = op->setBackOffsetsAfterDeltas(s, op->value());
			op->setDirty(true);
			l("before op propagate dirty");
			graph->nodePropagateDirty(op->index);
		}
	}

	return s;
}

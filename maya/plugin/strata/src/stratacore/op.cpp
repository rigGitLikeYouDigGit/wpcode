

#include "op.h"
#include "opgraph.h"

using namespace ed;

Status StrataOp::eval(StrataManifold& value,
	EvalAuxData* auxData, Status& s) {
	LOG("STRATAOP EVAL on node " + name + " - THIS IS WRONG");
	return s;
}


Status& StrataOp::gatherBackDeltas(Status& s, StrataManifold& finalManifold, SAtomBackDeltaGroup& result) {
	/* get initial deltas from end node
	*/
	return s;
}

/* assume this will run parallel between nodes - each object should only know immediate components to affect,
and result will only be components taken in by this node*/
SAtomBackDeltaGroup StrataOp::bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front) {
	/* pass in deltas to match -

	we work out what INPUTS (if any) would best match the given targets-
	return new AtomDeltaGroup representing that.

	SAVE initial target deltas on this op to match later

	those deltas may not be able to be matched exactly -
	subsequent method builds final offsets on this node, from previous node's best effort
	*/

	// save deltas on this node
	LOG("OP base fitBackDeltas:" + name);
	backDeltasToMatch = front;
	SAtomBackDeltaGroup result(front); // copy so we pass through any other elements
	// erase any elements created by this node from result
	for (int i : elements) {
		auto name = finalManifold.getEl(i)->name;
		auto found = result.targetMap.find(name);
		if (found != result.targetMap.end()) {
			result.targetMap.erase(found->first);
		}
	}
	return result;
}


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
	l("before backprop manifold: " + graph->results[graph->getOutputIndex()].printInfo());

	/* backwards pass, getting target deltas for elements/ nodes to match
	*/
	l("back prop backwards pass");
	for (int i = 0; i < static_cast<int>(generations.size()); i++) {
		l("back visit generation: " + str(i));

		std::vector<int>& toVisit = generations[i];
		std::vector<SAtomBackDeltaGroup> resultDeltas(toVisit.size()); // results of this iteration of back-prop
		for (int n = 0; n < static_cast<int>(toVisit.size()); n++) { // parallel this
			
			StrataOp* op = graph->getNode<StrataOp>(toVisit[n]);
			l("back visit node:" + op->name);
			resultDeltas[n] = op->bestFitBackDeltas(&s, finalManifold, deltaGrp);
			CRMSG(s, "error running back propagation on node " + op->name);
			l("op result after best fit: " + op->value().printInfo());
			op->setDirty(true);
			l("before op propagate dirty: " + op->name);
			graph->nodePropagateDirty(op->index);
		}

		// collate delta fronts
		for (int n = 1; n < static_cast<int>(resultDeltas.size()); n++) {
			resultDeltas[0].mergeOther(resultDeltas[n]);
		}
		deltaGrp = resultDeltas[0]; // need to do a copy here because we create the intermediate vals in the loop scope
		l("deltas after generation: " + deltaGrp.strInfo());
	}
	
	///* eval */
	//s = graph->evalGraph(s, fromNode->index, &auxData);
	return s;
}



#include "opgraph.h"
#include "op.h"
#include "../logger.h"

using namespace ed;
inline void StrataOpGraph::copyOtherNodesVector(const StrataOpGraph& other) {
	// function to deep-copy all nodes in the given vector from the argument graph
	LOG("opGraph copy other nodes");
	nodes.clear();
	nodes.reserve(other.nodes.size());
	for (auto& ptr : other.nodes) {
		//nodes.push_back(std::unique_ptr<DirtyNode>(ptr.get()->clone()));
		nodes.push_back(ptr.get()->clone<StrataOp>());
	}
}

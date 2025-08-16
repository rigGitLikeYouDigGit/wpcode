
#include <utility>
#include "opgraph.h"
#include "op.h"
#include "../logger.h"

using namespace strata;
void StrataOpGraph::copyOtherNodesVector(const StrataOpGraph& other) {
	// function to deep-copy all nodes in the given vector from the argument graph
	LOG("OpGraph COPY OTHER NODES");
	nodes.clear();
	nodes.reserve(other.nodes.size());
	for (auto& ptr : other.nodes) {
		//nodes.push_back(std::unique_ptr<DirtyNode>(ptr.get()->clone()));
		nodes.push_back(std::move( ptr.get()->clone() ) );
		//nodes.back()->graphPtr = this;
	}
	for (auto& ptr : nodes) {
		ptr->graphPtr = this;
	}
}

//StrataOpGraph* StrataOpGraph::clone_impl(bool copyAllResults) const {
//	auto newPtr = new StrataOpGraph(*this);
//	newPtr->copyOther(*this, copyAllResults);
//	return newPtr;
//};

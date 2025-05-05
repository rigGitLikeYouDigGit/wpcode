

#include "dirtyGraph.h"
using namespace ed;


std::map<std::string, DirtyNode*> DirtyNode::nameInputNodeMap() {
	std::map<std::string, DirtyNode*> result;
	for (auto i : inputs) {
		DirtyNode* node = graphPtr->getNode(i);
		result[node->name] = node;
	}
	return result;
}

std::vector<DirtyNode*> DirtyNode::inputNodes() {
	std::vector<DirtyNode*> result(inputs.size());
	for (auto i : inputs) {
		DirtyNode* node = graphPtr->getNode(i);
		result.push_back(node);
	}
	return result;
}

std::vector<std::string> DirtyNode::inputNames() {
	std::vector<std::string> result(inputs.size());
	for (auto i : inputs) {
		result.push_back(graphPtr->nodes[i].get()->name);
	}
	return result;
}

void DirtyNode::nodeError(std::string errorMsg) {
	if (!graphPtr) {
		return;
	}
	graphPtr->addNodeError(index, errorMsg);
}
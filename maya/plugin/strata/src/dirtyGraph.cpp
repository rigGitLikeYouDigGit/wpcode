

#include "dirtyGraph.h"
using namespace strata;


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


void DirtyGraph::nodePropagateDirty(int opIndex) {
	/* manually set dirty flags on node before calling this function -
	all flags set dirty will be propagated to nodes in future*/
	// propagate dirty stuff forwards to all nodes
	// we add bools to the base state - 
	// can't use false arguments to set clean with this function
	//LOG("node propagate dirty: " + strata::str(opIndex));
	DirtyNode* seedNode = getNode(opIndex);

	// if all flags match in dirty map, we consider this node has been visited, and skip tree
	Iterator it = iterNodes(
		seedNode,
		true, // forwards 
		true, // depth first
		false // include self
	);
	//for (auto nextNode : it) {
	while (it != it.end()) {
		if (*it == nullptr) {
			//l("iterator reached end null");
			break;
		}
		DirtyNode* nextNode = *it;
		if (nextNode->index == seedNode->index) { // skip the same node
			it++;
			continue;
		}
		
		//l("check node: " + nextNode->name);
		bool allMatch = true;
		for (auto pair : seedNode->dirtyMap) {
			if (!(nextNode->dirtyMap.count(pair.first))) { // propagate if not found
				allMatch = false;
				nextNode->dirtyMap[pair.first] = pair.second;
				continue;
			}
			if (nextNode->dirtyMap[pair.first] != pair.second) {
				allMatch = false;
				nextNode->dirtyMap[pair.first] = (nextNode->dirtyMap[pair.first] || pair.second); // if either is dirty, keep that value
			}
		}
		if (allMatch) { // all dirty flags match on this node, skip this tree
			it.skipTree();
		}
		it++;
	}
}



int DirtyGraph::mergeOther(const DirtyGraph& other, Status& s) {
	/* copy nodes from other not found in this graph -
	works by NAME ONLY
	indices of merged nodes WILL BE DIFFERENT
	we also pull in any edge connections from them

	LATER add in some way to check / add from only nodes in the critical path of the output
	(but with this system, that's guaranteed to be all nodes in the graph?)

	returns FIRST INDEX of added nodes, or -1 if none have been added
	*/
	//LOG("dirtyGraph mergeOther");
	int resultIndex = -1;
	NodeDelta nDelta = getNodeDelta(other);
	// copy nodes to add (required for correct edge delta
	for (std::string nodeName : nDelta.nodesToAdd) {
		//l("adding other node:" + nodeName);
		if (nodeName.empty()) {
			//l("other had an empty node name, exiting");
			return -1;
		}
		//importNode(other, other.nameIndexMap.find(nodeName))
		int otherIndex = other.nameIndexMap.at(nodeName);
		DirtyNode* importedNode = importNode(other, otherIndex);
		//l("dirtyGraph imported node is in this graph: " + str(importedNode->graphPtr == this));
		if (resultIndex == -1) {
			resultIndex = importedNode->index;
		}
		/* check that imported node has a name*/
		if (importedNode->name.empty()) {
			//l("imported node name is empty, exiting");
			return -1;
		}
	}
	EdgeDelta eDelta = getEdgeDelta(other, nDelta.nodesToAdd);

	// node inputs need to be re-indexed based on names
	for (auto& nodeInputsPair : eDelta.edgesToAdd) {
		DirtyNode* thisNode = getNode(nodeInputsPair.first);
		thisNode->inputs.clear();
		thisNode->inputs.reserve(eDelta.edgesToAdd[thisNode->name].size());
		for (auto inputNodeName : eDelta.edgesToAdd[thisNode->name]) {
			thisNode->inputs.push_back(nameIndexMap[inputNodeName]);
		}
		nodeInputsChanged(thisNode->index);

	}
	graphChanged = true;
	return resultIndex;
}
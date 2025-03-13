#pragma once

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <stack>
#include <deque>
#include <string>
#include <memory>
#include <initializer_list>
#include <iterator> // For std::forward_iterator_tag
#include <cstddef>  // For std::ptrdiff_t

#include "status.h"

#include "iterator.h"

/* abstracting pure topo structure of dependency graph - 
assuming that all nodes have a global unique name
(but disregard this if not useful)

we assume a single node is only valid for and belongs to a single type of graph?
ignoring values / evaluation for now
*/

namespace ed {

	//template<typename VT = int>
	struct DirtyGraph;
	//struct EvalGraph<>;

	// override flag type in node type to control how many dirty flags?


	struct DirtyNode {
		const int index;
		const std::string name;
		const DirtyGraph* graph;
		std::vector<int> inputs; // manage connections yourself

		// my python is showing - we use a map here for easier extensibility
		std::map<const std::string, bool> dirtyMap = { {"main" , true} };

		int temp_inDegree = 0;
		int temp_generation = 0;

		//DirtyNode* depthFirstConnections

		DirtyNode(const int index, const std::string name) : index(index), name(name) {}

		//std::unique_ptr<VT> result;

		virtual void reset() {
			// free memory owning result object
			//result.reset();
		}
	};


	//template<typename VT=int> // leave blank if you don't need complex evaluation
	struct DirtyGraph {
		/* abstracted graph behaviour for topological generations,
		dirty propagation etc
		losing efficiency with indices and names but it's fine

		most things here just stored in arrays


		can we somehow separate the evaluation behaviour from the raw topological graph?

		*/


		std::vector<std::unique_ptr<DirtyNode>> nodes;
		std::unordered_map<std::string, int> nameIndexMap;
		//std::vector<VT> results;

		bool graphChanged = true; // if true, needs full rebuild of topo generations
		// topo maps
		std::vector<std::unordered_set<int>> generations;
		// transient map for topology
		std::unordered_map<int, std::unordered_set<int>> nodeDirectDependentsMap;
		std::unordered_map<int, std::unordered_set<int>> nodeAllDependentsMap;

		template <typename NodeT = DirtyNode>
		const NodeT* addNode(const std::string& name) {
			/* create new node object,
			returns a new pointer to it*/

			const int newIndex = static_cast<int>(nodes.size());
			//auto il = { newIndex, name };
			//auto ptr = std::make_unique<NodeT>(newIndex, name);
			//auto ptr = std::make_unique<NodeT>(il);
			nodes.push_back(
				std::move(
					std::make_unique<NodeT>(newIndex, name)
					//ptr
				)
			);
			//std::unique_ptr<NodeT> newNodePtr = nodes[newIndex];
			const NodeT* newNodePtr = nodes[newIndex].get();
			nameIndexMap[newNodePtr->name] = newIndex;
			graphChanged = true;
			return newNodePtr;
		}

		inline DirtyNode* getNode(DirtyNode*& node) {
			// included here for similar syntax to get op pointer, no matter the input
			// not sure if this is actually useful in c++
			return node;
		}
		inline DirtyNode* getNode(const int& index) {
			return nodes[index].get();
		}
		inline DirtyNode* getNode(const std::string& nodeName) {
			return nodes[nameIndexMap[nodeName]].get();
		}

		Status getTopologicalGenerations(std::vector<std::unordered_set<int>>& result, Status& s) {
			/* get sets of nodes guaranteed not to depend on each other
			// use for priority in scheduling work, but too restrictive to rely
			// on totally for execution
			( eval-ing generations one-by-one means each will always be waiting on the last node in that generation, which might not be useful)
			*/
			std::unordered_set<int> zeroDegree;
			nodeDirectDependentsMap.clear();
			nodeAllDependentsMap.clear();

			// first gather all nodes with no inputs
			for (auto& node : nodes) {
				node->temp_inDegree = static_cast<int>(node->inputs.size());
				// build topology outputs
				for (int inputId : node->inputs) {
					// add this node to the dependents of all its inputs
					nodeDirectDependentsMap[inputId].insert(node->index);
				}
				// if node has no inputs, put it in first generation
				if (!node->inputs.size()) {
					zeroDegree.insert(node->index);
				}
			}
			/* build generations
			*
			*/
			while (zeroDegree.size()) {
				result.push_back(zeroDegree); // add all zero degree generation to result
				zeroDegree.clear();

				// check over all direct children of current generation
				for (auto& nodeId : result[-1]) {
					// can parallelise this
					DirtyNode* node = getNode(nodeId);
					// all outputs of this node - decrement their inDegree by 1
					for (int dependentId : nodeDirectDependentsMap[node->index]) {
						DirtyNode* dependent = getNode(dependentId);
						dependent->temp_inDegree -= 1;
						// if no in_degree left, add this node to the next generation
						if (dependent->temp_inDegree < 1) {
							zeroDegree.insert(dependent->index);
						}
					}
				}
			}
			// set generation ids on all nodes
			int i = 0;
			for (auto& nodeIdSet : result) {
				for (auto& nodeId : nodeIdSet) {
					DirtyNode* node = getNode(nodeId);
					node->temp_generation = i;
				}
				i += 1;
			}
			//return result;
			return s;
		}


		Status rebuildGraphStructure(Status& s) {
			/* rebuild graph tables and caches from current ops in vector*/
			nameIndexMap.clear();
			nameIndexMap.reserve(nodes.size());
			for (size_t i = 0; i < nodes.size(); i++) {
				nameIndexMap[nodes[i]->name] = static_cast<int>(i);
			}
			std::vector<std::unordered_set<int>> result;
			s = getTopologicalGenerations(result, s);
			graphChanged = false;
			return s;
		}


		inline std::unordered_set<int>* nodeOutputs(const int opIndex) {
			if (graphChanged) {
				Status s;
				rebuildGraphStructure(s);
			}
			return &nodeDirectDependentsMap[opIndex];
		}

		inline std::unordered_set<int>* nodeOutputs(const DirtyNode* node) {
			return nodeOutputs(node->index);
		}

		/* functions to get nodes in history and future,
		TODO: maybe cache these? but they only matter in graph editing,
		I don't think there's any point in it
		*/
		std::unordered_set<int> nodesInHistory(int opIndex) {
			/* flat set of all nodes found in history*/
			std::unordered_set<int> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					DirtyNode* checkOp = getNode(checkNodeIndex);
					newToCheck.insert(checkOp->inputs.begin(), checkOp->inputs.end());

					// add the inputs to result this iteration to
					result.insert(checkOp->inputs.begin(), checkOp->inputs.end());
				}
				toCheck = newToCheck;
			}
			return result;
		}

		SmallList<std::unordered_set<int>, 8> nodesInHistory(int opIndex, bool returnGenerations) {
			/* generation list of nodes in history*/
			SmallList<std::unordered_set<int>, 8> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					DirtyNode* checkOp = getNode(checkNodeIndex);
					newToCheck.insert(checkOp->inputs.begin(), checkOp->inputs.end());
				}
				// add the inputs to this generation's result
				result.push_back(
					//std::unordered_set<int>(newToCheck.begin(), newToCheck.end())
					newToCheck
				);
				toCheck = newToCheck;
			}
			return result;
		}

		std::unordered_set<int> nodesInFuture(int opIndex) {
			/* flat set of all nodes found in future*/
			std::unordered_set<int> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					std::unordered_set<int>* outputIds = nodeOutputs(checkNodeIndex);
					newToCheck.insert(outputIds->begin(), outputIds->end());

					// add the outputs to the result of this iteration
					result.insert(outputIds->begin(), outputIds->end());
				}
				toCheck = newToCheck;
			}
			return result;
		}

		SmallList<std::unordered_set<int>, 8> nodesInFuture(int opIndex, bool returnGenerations) {
			/* generation list of nodes in future*/
			SmallList<std::unordered_set<int>, 8> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					std::unordered_set<int>* outputIds = nodeOutputs(checkNodeIndex);
					newToCheck.insert(outputIds->begin(), outputIds->end());
				}
				// add the inputs to result this iteration to
				result.push_back(
					//std::unordered_set<int>(newToCheck.begin(), newToCheck.end())
					newToCheck
				);
				toCheck = newToCheck;
			}
			return result;
		}

		struct Iterator {
			// from internalPointers.com - thanks a ton
			using iterator_category = std::forward_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using value_type = DirtyNode*;
			using pointer = DirtyNode*;  // or also value_type*
			//using reference = valueType&;  // or also value_type&
			using reference = DirtyNode*;  // or also value_type&

			pointer nodePtr;
			DirtyGraph* itGraph;
			bool forwards = true;
			bool depthFirst = true;
			bool includeSelf = true;

			pointer _origNode;

			std::stack<DirtyNode*> nodeStack;
			std::unordered_set<DirtyNode*> discoveredSet;

			//std::stack<DirtyNode*> parentStack;



			Iterator(pointer ptr, DirtyGraph* itGraph, bool forwards, bool depthFirst, bool includeSelf) :
				nodePtr(ptr), itGraph(itGraph),
				_origNode(ptr), // set original node pointer to start 
				forwards(forwards), depthFirst(depthFirst), includeSelf(includeSelf) {

				if (discoveredSet.size() == 0) {
					nodeStack.push(nodePtr);
				}

				// by default the iterator starts at the source node - if we don't want to include it,
				// we do one step on init
				if (!includeSelf) {
					doStep();
				}
			}

			// second copier lets you pass in stack and set during iteration
			Iterator(pointer ptr, DirtyGraph* itGraph, bool forwards, bool depthFirst, bool includeSelf,
				std::stack<DirtyNode*>& nodeStack, std::unordered_set<DirtyNode*>& discoveredSet) :
				nodePtr(ptr), itGraph(itGraph),
				forwards(forwards), depthFirst(depthFirst), includeSelf(includeSelf),
				nodeStack{ nodeStack }, discoveredSet{ discoveredSet }
			{
				if (discoveredSet.size() == 0) {
					nodeStack.push(nodePtr);
				}

				if (!includeSelf) {
					doStep();
				}
			}


			reference operator*() const { return nodePtr; }
			pointer operator->() { return nodePtr; }

			//void dfs() { // basic process, sadly this isn't python so 
			//	if (discoveredSet.size() == 0) {
			//		nodeStack.push(nodePtr);
			//		////if (includeSelf) { // return the source node if it's the first iteration
			//		////	yield *this;
			//		////}
			//		//if (!includeSelf) {
			//		//	discoveredSet.insert()
			//		//}
			//	}
			//	while (nodeStack.size()) {
			//		DirtyNode* ptr = nodeStack.top();
			//		nodeStack.pop();
			//		if (!discoveredSet.count(ptr)) {

			//			// if the set has stuff in it OR if it's empty, and we includeSelf, yield
			//			if (discoveredSet.size() || includeSelf) {
			//				yield ptr;
			//			}
			//			discoveredSet.insert(ptr);
			//			for (int in : ptr->inputs) {
			//				nodeStack.push(itGraph->getNode(in));
			//			}
			//		}
			//	}
			//}

			void doStep() {
				// do single step to move iterator forwards
				// this is almost all we do in increment function
							// check if the iterator has just been created 

				if (forwards) {// backwards is tracked explicitly by nodes, don't need full graph rebuild
					// if forwards, we need to check graph for rebuilds
					if (itGraph->graphChanged) {
						Status s;
						itGraph->rebuildGraphStructure(s);
					}
					// don't change graph structure during iteration please
				}

				// check through stack to find the last undiscovered node?
				while (discoveredSet.count(nodeStack.top()) && nodeStack.size()) {
					nodeStack.pop();
				}

				// if stack is empty, iterator has concluded
				// set ptr to null and return
				if (nodeStack.size() == 0) {
					nodePtr = nullptr;
					return;
				}

				// do one iteration of the dfs sketch above

				DirtyNode* ptr = nodeStack.top();
				nodeStack.pop();

				//if (!discoveredSet.count(ptr)) {

				// if the set has stuff in it OR if it's empty, and we includeSelf, yield
				discoveredSet.insert(ptr);
				if (forwards) {
					if (itGraph->nodeDirectDependentsMap.count(ptr->index)) {
						for (int in : itGraph->nodeDirectDependentsMap[ptr->index]) {
							nodeStack.push(itGraph->getNode(in));
						}
					}
				}
				else { // backwards just use direct references on node
					if (ptr->inputs.size()) {
					}
					for (int in : ptr->inputs) {
						nodeStack.push(itGraph->getNode(in));
					}
				}
				nodePtr = ptr;
				return;
				//}
			}

			void skipTree() {
				// if we want to prune a depth-first search early, 
				//we pop nodes from the stack til we get to the current node, and pop that too?
				// stacks are cool

				// to do this we need to LEAVE entries in the stack? 

				// or we just mark direct outputs of current node as discovered
				if (forwards) {
					if (itGraph->nodeDirectDependentsMap.count(nodePtr->index)) {
						for (int in : itGraph->nodeDirectDependentsMap[nodePtr->index]) {
							discoveredSet.insert(itGraph->getNode(in));
						}
					}
				}
				else { // backwards just use direct references on node
					for (int in : nodePtr->inputs) {
						discoveredSet.insert(itGraph->getNode(in));
					}
				}
			}

			// Prefix increment
			Iterator& operator++() {
				/* here we advance one step along nodes
				// for simplicity we just use one level of sets -
				maybe there's a better way that doesn't need sets, by tracking visited?
				maybe we can actually cache the topology links in graph build
				*/
				doStep();
				return *this;

			}

			// Postfix increment
			Iterator operator++(int) {
				Iterator tmp = *this;
				++(*this);
				return tmp;
			}

			friend bool operator== (const Iterator& a, const Iterator& b) { return a.nodePtr == b.nodePtr; };
			friend bool operator!= (const Iterator& a, const Iterator& b) { return a.nodePtr != b.nodePtr; };

			Iterator begin() {
				return Iterator(_origNode, itGraph,
					forwards, depthFirst, includeSelf);
			} // either direction begins at the original node
			Iterator end() {
				return Iterator(nullptr, itGraph,
					forwards, depthFirst, includeSelf);
			} // 200 is out of bounds
		};

		Iterator iterNodes(DirtyNode* sourceNode, bool forwards = true, bool depthFirst = true, bool includeSelf = true) {
			// build and return an iterator set up in this way
			return Iterator(sourceNode, this, forwards, depthFirst, includeSelf);
		}

		void nodePropagateDirty(int opIndex) {
			/* manually set dirty flags on node before calling this function -
			all flags set dirty will be propagated to nodes in future*/
			// propagate dirty stuff forwards to all nodes
			// we add bools to the base state - 
			// can't use false arguments to set clean with this function
			DirtyNode* seedNode = getNode(opIndex);

			// if all flags match in dirty map, we consider this node has been visited, and skip tree
			Iterator it = iterNodes(seedNode, true, true, false);
			//for (auto nextNode : it) {
			while (it != it.end()) {
				DirtyNode* nextNode = *it;
				bool allMatch = true;
				for (auto pair : seedNode->dirtyMap) {
					if (!(nextNode->dirtyMap.count(pair.first))) { // propagate if not found
						allMatch = false;
						nextNode->dirtyMap[pair.first] = pair.second;
						continue;
					}
					if (nextNode->dirtyMap[pair.first] != pair.second) {
						allMatch = false;
						nextNode->dirtyMap[pair.first] += pair.second; // if either is dirty, keep that value
					}
				}
				if (allMatch) { // all dirty flags match on this node, skip this tree
					it.skipTree();
				}
			}
		};
	};
	template <typename VT = int>
	struct EvalGraph : DirtyGraph {
		// adding mechanisms for evaluation and caching results

		std::vector<VT> results;

		template <typename NodeT = DirtyNode>
		const NodeT* addNode(const std::string& name, VT defaultValue = nullptr) {
			const NodeT* baseResult = DirtyGraph::addNode(name);
			if (defaultValue == nullptr) {
				defaultValue = VT();
			}
			results.push_back(defaultValue);
			return baseResult;
		}
	};

}
	//static DirtyGraph testGraph;

	//static auto newNode = testGraph.addNode<>("newNode");




		//	void nodeInputsChanged(int opIndex) {
		//		graphChanged = true;
		//	}



		//	int checkLegalInput(int opIndex, int testDriverOpIndex) {
		//		/* if result is 0 all good
		//		// if not, error and it's illegal
		//		check for feedback loops
		//		*/
		//		if (opIndex == testDriverOpIndex) { // can't connect to itself
		//			return 1;
		//		}
		//		std::unordered_set<int> futureOps = opsInFuture(opIndex);
		//		if (futureOps.count(testDriverOpIndex)) {
		//			// node found in future, illegal feedback loop
		//			return 2;
		//		}
		//		return 0;
		//	}



		//	Status evalOp(StrataOp* op, Status& s) {
		//		/* encapsulated evaluation for a single op
		//		* we guarantee that all an op's input nodes will already
		//		* have been evaluated
		//		*/

		//		// reset node state
		//		op->reset();

		//		// copy geo into node's result, if it has inputs
		//		if (op->inputs.size()) {
		//			StrataOp* mainInput = getOp(op->inputs[0]);
		//			// copy the manifold result of previous node
		//			op->result = mainInput->result;
		//		}

		//		/*TODO: check for errors in evaluation, somehow -
		//		that would mean passing out errors to graph eval function

		//		...we could hypothetically CHECK an MSTATUS...
		//		...and hypothetically RETURN_IT...
		//		*/

		//		// if params have changed, recompile
		//		if (op->paramsDirty) {
		//			s = op->evalParams(op->rootExpNode, s);
		//			CRSTAT(s);
		//			op->paramsDirty = false;
		//		}

		//		// eval topology if dirty
		//		if (op->topoDirty) {
		//			s = (op->result, s);
		//			CRSTAT(s);
		//			op->topoDirty = false;
		//		}

		//		// eval data if dirty
		//		if (op->dataDirty) {
		//			s = op->evalData(op->result, s);
		//			CRSTAT(s);
		//			op->dataDirty = false;
		//		}

		//		// donezo
		//		return s;
		//	}

		//	Status evalOpGraph(Status& s, int upToNodeId = -1) {
		//		// TODO: don't eval topo and data all the time obviously
		//		// just for now

		//		/* first input of each node by default should be the manifold stream
		//		to write on. if a node has no input, it's an ancestor in the graph-
		//		create an empty manifold object for that object

		//		I don't know if there's an alternative to copying the manifold to every node's output?
		//		surely that's ridiculously wasteful.
		//		but using a constant reference means we can't have a "breakpoint" in the graph, since the same
		//		object will be edited constantly?

		//		until a node branches, we can work on only one object?

		//		copy everything for now. make it exist, then make it efficient.

		//		*/

		//		// sort topo generations in nodes if graph has changed
		//		if (graphChanged) {
		//			rebuildGraphStructure(s);

		//		}
		//		CRMSG(s, "ERROR rebuilding graph structure ahead of graph eval, halting Strata graph");

		//		bool foundEndNode = false;
		//		// for now just work in generations
		//		// go through generations in sequence
		//		for (auto& generation : generations) {

		//			// go through each node in generation
		//			// this is the bit that can be parallelised
		//			for (auto& nodeId : generation) {
		//				StrataOp* node = getOp(nodeId);
		//				// check if anything on node is dirty - if so, continue
		//				if (!(node->dataDirty + node->topoDirty + node->paramsDirty)) {
		//					continue;
		//				}
		//				// eval whole node
		//				evalOp(node, s);
		//				CRMSG(s, "ERROR eval-ing op " + node->name + ", halting Strata graph ");

		//				// mark outputs dirty, ensure they get eval'd in next generation
		//				for (int dependentId : nodeDependentsMap[node->index]) {
		//					StrataOp* dependent = getOp(dependentId);
		//					dependent->topoDirty = true;
		//				}
		//				if (node->index == upToNodeId) {
		//					foundEndNode = true;
		//					break;
		//				}
		//			}
		//			// stop all eval if end node reached
		//			if (foundEndNode) {
		//				break;
		//			}
		//		}
		//		return s;
		//	}

		//};

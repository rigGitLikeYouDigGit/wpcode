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
#include "macro.h"
#include "lib.h"
#include "name.h"
#include "../containers.h"

#include "iterator.h"

#include "mixin.h"
#include "logger.h"

/* abstracting pure topo structure of dependency graph - 
assuming that all nodes have a global unique name
(but disregard this if not useful)

we assume a single node is only valid for and belongs to a single type of graph?
ignoring values / evaluation for now


we can't avoid value templating for a node included in an evaluation graph

*/

namespace strata {

	/*template <typename T>
	inline std::set<T> setIntersect(
		std::set<T>& b,
		std::set<T>& b,
		)*/

	//template<typename VT = int>
	//template <typename NodeT>
	struct DirtyGraph;
	template <typename VT>
	struct EvalGraph;

	// ============================================================================
	// DirtyNode - Now templated on graph type
	// ============================================================================
	struct DirtyNode {
		int index = -1;
		StrataName name = "_";
		void* graphPtr = nullptr;

		template <typename GraphT = DirtyGraph>
		inline GraphT* graph() { return static_cast<GraphT*>(graphPtr); }

		SmallList<ushort> inputs = {}; // manage connections yourself
		bool enabled = true; /* should this be per-node, or a set on the parent object?*/
		bool dirty = true; /* this is a transient state, not serialised - if true, needs to be re-evaluated*/

		int temp_inDegree = 0;
		int temp_generation = 0;

		DirtyNode() {}
		DirtyNode(int index, StrataName name) : index(index), name(name) {}
		~DirtyNode() {}

		inline bool setDirty(bool state) {
			/* returns the previously set dirty state*/
			bool oldState = dirty;
			dirty = state;
			return oldState;
		}

		//inline std::map<std::string, DirtyNode*> nameInputNodeMap();

		//std::vector<DirtyNode*> inputNodes();

		//std::vector<std::string> inputNames();


		Status postConstructor() {
			/* called after node added to graph, all connections set up*/
			return Status();
		}

		//inline void nodeError(std::string errorMsg);
	};

	//// DirtyNode inline method implementations
	//inline std::map<std::string, DirtyNode*> DirtyNode::nameInputNodeMap() {
	//	std::map<std::string, DirtyNode*> result;
	//	for (auto i : inputs) {
	//		DirtyNode* node = graph()->getNode(i);
	//		result[node->name] = node;
	//	}
	//	return result;
	//}

	//inline std::vector<DirtyNode*> DirtyNode::inputNodes() {
	//	std::vector<DirtyNode*> result(inputs.size());
	//	for (auto i : inputs) {
	//		DirtyNode* node = graph()->getNode(i);
	//		result.push_back(node);
	//	}
	//	return result;
	//}

	//inline std::vector<std::string> DirtyNode::inputNames() {
	//	std::vector<std::string> result(inputs.size());
	//	for (auto i : inputs) {
	//		result.push_back(graph()->getNode(i)->name);
	//	}
	//	return result;
	//}

	//inline void DirtyNode::nodeError(std::string errorMsg) {
	//	if (!graphPtr) {
	//		return;
	//	}
	//	graph()->addNodeError(index, errorMsg);
	//}

	/* CRTP for graph to control the node type stored on the base */
	template <typename Derived, typename NodeT = DirtyNode>
	struct DirtyGraphBase {
		/* abstracted graph behaviour for topological generations,
		dirty propagation etc
		losing efficiency with indices and names but it's fine

		most things here just stored in arrays

		can we somehow separate the evaluation behaviour from the raw topological graph?

		*/

		/* MEMBERS TO COPY / SERIALISE */
		std::vector<NodeT> nodes;

		std::unordered_map<StrataName, int> nameIndexMap;
		int _outputIndex = -1; // houdini-esque tracking which node is 
		// designated as the output of the graph


		/* TRANSIENT MEMBERS */
		bool graphChanged = true; // if true, needs full rebuild of topo generations
		// topo maps
		std::vector<std::vector<int>> generations = { {} };
		// transient map for topology
		std::unordered_map<int, std::vector<int>> nodeDirectDependentsMap;
		std::unordered_map<int, std::vector<int>> nodeAllDependentsMap;

		// map to track nodes that have errored
		std::map<int, std::string> indexErrorMap;

		DirtyGraphBase() {}
		~DirtyGraphBase() = default;
		DirtyGraphBase(DirtyGraphBase const& other) {
			copyOther(other);
		}
		DirtyGraphBase(DirtyGraphBase&& other) = default;
		DirtyGraphBase& operator=(DirtyGraphBase const& other) {
			copyOther(other);
		}
		DirtyGraphBase& operator=(DirtyGraphBase&& other) = default;


		///////////////// COPYING ////////

		template<typename otherGraphT>
		void copyOther(otherGraphT& other) {
			nodes.resize(other.nodes.size());
			for (size_t i = 0; i < other.nodes.size(); i++) {
				nodes[i] = other.nodes[i];
				nodes[i].graphPtr = this;
			}
		}

		Status setNodeName(NodeT* nodePtr, std::string newName) {
			// the one concession to mutable state for now in graphs
			Status s;
			int index = nameIndexMap[nodePtr->name];
			nameIndexMap.erase(nodePtr->name);
			//nameIndexMap[newName] = index;
			nameIndexMap.insert({ std::string(newName), index });
			nodePtr->name = newName;
			return s;
		}


		/*template <typename NodeT = DirtyNode>*/
		NodeT* addNode(const std::string& name, bool _callPostConstructor = true) {
			/* create new node object,
			returns a new pointer to it.

			callPostConstructor true by default -
			if a derived graph type calls this, set to false to call post
			only at the end of the overridden function

			should we convert all this to return status code?
			*/
			//if (nameIndexMap.count(name) != 0) {
			if (nameIndexMap.find(name) != nameIndexMap.end()) {
				Status s;
				CWMSG(s, "Name " + name + " already found in dirty graph, returning nullptr");
				return nullptr;
			}

			const int newIndex = static_cast<int>(nodes.size());
			/*auto nodePtr = std::make_unique<NodeT>(newIndex, name);
			nodes.push_back(std::move(nodePtr));*/
			nodes.push_back(NodeT(newIndex, name));


			NodeT* newNodePtr = &nodes[newIndex];
			//nameIndexMap[std::string(newNodePtr->name)] = newIndex;
			nameIndexMap.insert({ newNodePtr->name, newIndex });

			newNodePtr->graphPtr = this;
			graphChanged = true;

			if (_callPostConstructor) {
				Status s = newNodePtr->postConstructor();
				CWMSG(s, "post-constructor on node " + newNodePtr->name + " failed!")

			}
			return newNodePtr;
		}

		inline NodeT* getNode(DirtyNode*& node) const {
			// included here for similar syntax to get op pointer, no matter the input
			// not sure if this is actually useful in c++
			return node;
		}
		inline NodeT* getNode(const int& index) const {
			//DEBUGSL("get node by index " + std::to_string(index));
			//DEBUGSL("nodes size: " + std::to_string(nodes.size()));
			if (nodes.size() <= index) {
				return nullptr;
			}
			return &nodes[index];
		}
		inline NodeT* getNode(const std::string& nodeName) const {

			auto check = nameIndexMap.find(nodeName);
			if (check == nameIndexMap.end()) { return nullptr; }
			return &nodes[check->second];
		}

		//template<typename NodeT>
		//inline NodeT* getNode(DirtyNode*& node) const {
		//	return dynamic_cast<NodeT*>(getNode(node));
		//}
		//template<typename NodeT>
		//inline NodeT* getNode(const int& index) const {
		//	return dynamic_cast<NodeT*>(getNode(index));
		//}
		////template<typename NodeT>
		//inline NodeT* getNode(const std::string& nodeName) const {
		//	return dynamic_cast<NodeT*>(getNode(nodeName));
		//}

		template<typename argT, unsigned int N=16>
		inline SmallList<NodeT*, N> getNodes(argT* start, argT* end) {
			/* iterator compatibility*/
			SmallList<NodeT*, N=16u> result;
			while (start != end) {
				result.push_back(getNode(*start));
				start++;
			}
			return result;
		}


		//virtual Status postConstructor() {
		//	/* called after node added to graph, all connections set up*/
		//	return Status();
		//}


		inline void addNodeError(int index, std::string& errorMsg) {
			indexErrorMap.insert(std::make_pair(index, errorMsg)
			);
		}

		const inline StrataName outputNodeName() {
			return nodes[getOutputIndex()].name;
		}

		inline bool hasOutputNode() {
			return (nodes.size() && (_outputIndex > -1));
		}

		inline void setOutputNode(int index) {
			_outputIndex = index;
			graphChanged = true;
		}
		inline int getOutputIndex() const {
			/* return the output if set,
			or just the last node added if -1*/
			if (_outputIndex < 0) {
				return static_cast<int>(nodes.size() - 1);
			}
			return _outputIndex;
		}
		/*template<typename seqT>
		inline seqT<int> namesToIndices(seqT<str> names) {
		}*/
		// couldn't work out how to template on a template like this

		inline SmallList<int> namesToIndices(std::string* start, std::string* end) {
			SmallList<int> result;
			while (start != end) {
				result.push_back(nameIndexMap[*start]);
				start++;
			}
			return result;
		}
		inline SmallList<StrataName> indicesToNames(int* start, int* end) {
			SmallList<StrataName> result;
			while (start != end) {
				result.push_back(nodes[*start].name);
				start++;
			}
			return result;
		}


		Status getTopologicalGenerations(std::vector<std::vector<int>>& result, Status& s) {
			/* get sets of nodes guaranteed not to depend on each other
			// use for priority in scheduling work, but too restrictive to rely
			// on totally for execution
			( eval-ing generations one-by-one means each will always be waiting on the last node in that generation, which might not be useful)
			*/
			std::vector<int> zeroDegree;
			nodeDirectDependentsMap.clear();
			nodeAllDependentsMap.clear();
			LOG("gather nodes no inputs, n nodes:" + str(nodes.size()));
			// first gather all nodes with no inputs
			for (auto& node : nodes) {
				l("check node " + node.name + " inputs: ");
				DEBUGVI(node.inputs);
				node.temp_inDegree = static_cast<int>(node.inputs.size());
				// build topology outputs
				int inputCount = 0;
				for (int inputId : node.inputs) {
					if (inputId < 0) {
						continue;
					}
					// add this node to the dependents of all its inputs
					nodeDirectDependentsMap[inputId].push_back(node.index);
					inputCount += 1;
				}
				// if node has no inputs, put it in first generation
				if (!inputCount) {
					zeroDegree.push_back(node.index);
				}
			}
			l("gathered seed nodes:");
			DEBUGVI(zeroDegree);
			/* build generations
			*
			*/

			while (zeroDegree.size()) {
				l("seed outer iter, zdegree:" + std::to_string(zeroDegree.size()) + "result: " + std::to_string(result.size()));
				result.push_back(zeroDegree); // add all zero degree generation to result
				zeroDegree.clear();

				// check over all direct children of current generation
				for (auto nodeId : result[seqIndex(-1, result.size())]) {
					// can parallelise this
					DirtyNode* node = getNode(nodeId);
					// all outputs of this node - decrement their inDegree by 1
					for (int dependentId : nodeDirectDependentsMap[node->index]) {
						DirtyNode* dependent = getNode(dependentId);
						dependent->temp_inDegree -= 1;
						// if no in_degree left, add this node to the next generation
						if (dependent->temp_inDegree < 1) {
							zeroDegree.push_back(int(dependent->index));
						}
					}
				}
			}
			l("built generations");
			// set generation ids on all nodes
			int i = 0;
			for (auto& nodeIdSet : result) {
				for (auto& nodeId : nodeIdSet) {
					DirtyNode* node = getNode(nodeId);
					node->temp_generation = i;
				}
				i += 1;
			}
			generations = result;
			l("set generation ids");
			//return result;
			return s;
		}


		Status rebuildGraphStructure(Status& s) {
			/* rebuild graph tables and caches from current ops in vector*/

			LOG("clearing if size:" + str(nameIndexMap.size()));
			if (nameIndexMap.size()) {
				l("rebuild, nodes are:");

				for (auto p : nameIndexMap) {
					l(p.first + str(p.second));
				}
				nameIndexMap.clear();
			}

			//int nodesSize = static_cast<int>(nodes.size());
			//nameIndexMap.reserve(nodesSize);
			for (size_t i = 0; i < nodes.size(); i++) {
				std::string nodeName(nodes[i].name);
				nameIndexMap[nodeName] = static_cast<int>(i);
			}
			std::vector<std::vector<int>> result;
			s = getTopologicalGenerations(result, s);
			graphChanged = false;
			return s;
		}


		inline void syncGraphStructure() {
			if (graphChanged) {
				Status s;
				rebuildGraphStructure(s);
			}
		}

		inline std::vector<int>* nodeOutputs(const int opIndex) {
			if (graphChanged) {
				Status s;
				rebuildGraphStructure(s);
			}
			return &nodeDirectDependentsMap[opIndex];
		}

		inline std::vector<int>* nodeOutputs(const DirtyNode* node) {
			return nodeOutputs(node->index);
		}

		/* functions to get nodes in history and future,
		TODO: maybe cache these? but they only matter in graph editing,
		I don't think there's any point in it
		*/
		//std::unordered_set<int> nodesInHistory(int opIndex) {
		std::vector<int> nodesInHistory(int opIndex) {
			/* flat set of all nodes found in history
			we also include final op index - if not desired, iterate only
			up to last value

			seed node is first, values increasing in reverse
			*/
			LOG("nodesInHistory flat " + std::to_string(opIndex));
			std::vector<int> result;

			std::vector<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::vector<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					if (checkNodeIndex < 0) {
						continue;
					}
					DirtyNode* checkOp = getNode(checkNodeIndex);

					//newToCheck.insert(checkOp->inputs.begin(), checkOp->inputs.end());
					//newToCheck.emplace_back(checkOp->inputs.begin(), checkOp->inputs.end());
					newToCheck.insert(newToCheck.end(), checkOp->inputs.begin(), checkOp->inputs.end());

					// add the inputs to result this iteration to
					//result.insert(checkNodeIndex);
					result.push_back(checkNodeIndex);
					//result.insert(checkOp->inputs.begin(), checkOp->inputs.end());
				}
				toCheck = newToCheck;
			}
			l("nodesInHistory result");
			DEBUGVI(result);
			return result;
		}

		std::vector<std::vector<int>> nodesInHistory(int opIndex, bool returnGenerations) {
			/* generation list of nodes in history
			return generations where latest nodes guaranteed to be included before deeper nodes

			*/
			LOG("nodesInHistory generations");
			std::unordered_map<int, int> visited; // {index : highest degree}
			std::stack<int> toCheck;
			toCheck.push(opIndex);

			visited[opIndex] = 0;

			int maxDepth = 0;
			while (toCheck.size()) {
				int checkIndex = toCheck.top();
				toCheck.pop();
				for (int inIndex : getNode(checkIndex)->inputs) {
					if (inIndex < 0) {
						continue;
					}
					auto found = visited.find(inIndex);
					if (found == visited.end()) {
						visited[inIndex] = 0;
						found = visited.find(inIndex);
					}
					int checkDepth = visited.find(checkIndex)->second + 1;
					int foundDepth = std::max<int>({ found->second, checkDepth });
					visited.insert_or_assign(inIndex, foundDepth);

					toCheck.push(inIndex);
					maxDepth = std::max(maxDepth, foundDepth);
				}
			}
			// sort into vector of vectors
			std::vector<std::vector<int>> result(maxDepth + 1);
			for (auto p : visited) {
				result[p.second].push_back(p.first);
			}
			return result;
		}

		std::vector<int> nodesInFuture(int opIndex) {
			/* flat set of all nodes found in future*/
			std::vector<int> result;

			std::vector<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::vector<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					std::vector<int>* outputIds = nodeOutputs(checkNodeIndex);
					newToCheck.insert(newToCheck.end(), outputIds->begin(), outputIds->end());

					// add the outputs to the result of this iteration
					result.insert(result.end(), outputIds->begin(), outputIds->end());
				}
				toCheck = newToCheck;
			}
			return result;
		}

		//SmallList<std::unordered_set<int>, 8> nodesInFuture(int opIndex, bool returnGenerations) {
		std::vector<std::vector<int>> nodesInFuture(int opIndex, bool returnGenerations) {
			/* generation list of nodes in future*/
			//SmallList<std::unordered_set<int>, 8> result;
			std::vector<std::vector<int>> result;

			std::vector<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::vector<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					std::vector<int>* outputIds = nodeOutputs(checkNodeIndex);
					newToCheck.insert(newToCheck.end(), outputIds->begin(), outputIds->end());
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

		int graphTreeRoot() {
			/* disgusting braindead function, I'll check for the correct way sometime
			if -1, graph is not a tree
			otherwise return the index of the final node in graph
			*/
			int result = -1;
			syncGraphStructure();
			//std::unordered_set<int> nodesToCheck(nodes.begin(), nodes.end());
			std::unordered_set<int> nodesToCheck(nodes.size());
			for (auto& node : nodes) {
				nodesToCheck.insert(node.index);
			}
			while (nodesToCheck.size()) {
				int testNode = *nodesToCheck.begin();
				auto testNodesHistory = nodesInHistory(testNode);
				// does this node include every other node in its history?
				if (testNodesHistory.size() == (nodes.size() - 1)) {
					return testNode;
				}
				nodesToCheck.erase(testNode);
			}
			return result;
		}

		
		struct Iterator {
			// from internalPointers.com - thanks a ton
			using iterator_category = std::forward_iterator_tag;
			using difference_type = std::ptrdiff_t;
			using value_type = NodeT*;
			using pointer = NodeT*;  // or also value_type*
			//using reference = valueType&;  // or also value_type&
			using reference = NodeT*;  // or also value_type&

			pointer nodePtr;
			//DirtyGraph* itGraph;
			DirtyGraphBase* itGraph;
			bool forwards = true;
			bool depthFirst = true;
			bool includeSelf = true;

			pointer _origNode;

			std::stack<NodeT*> nodeStack = {};
			std::unordered_set<NodeT*> discoveredSet = {};

			//std::stack<DirtyNode*> parentStack;



			Iterator(pointer ptr, DirtyGraphBase* itGraph, bool forwards, bool depthFirst, bool includeSelf) :
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
			Iterator(pointer ptr, DirtyGraphBase* itGraph, bool forwards, bool depthFirst, bool includeSelf,
				std::stack<NodeT*>& nodeStack, std::unordered_set<NodeT*>& discoveredSet) :
				nodePtr(ptr), _origNode(ptr),
				itGraph(itGraph),
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
								// if stack is empty, iterator has concluded
				// set ptr to null and return
				if (nodeStack.size() <= 0) {
					nodePtr = nullptr;
					return;
				}
				if (forwards) {// backwards is tracked explicitly by nodes, don't need full graph rebuild
					// if forwards, we need to check graph for rebuilds
					if (itGraph->graphChanged) {
						Status s;
						itGraph->rebuildGraphStructure(s);
					}
					// don't change graph structure during iteration please
				}


				// check through stack to find the last undiscovered node?
				while (nodeStack.size() && discoveredSet.count(nodeStack.top())) {
					nodeStack.pop();

				}

				// if stack is empty, iterator has concluded
				// set ptr to null and return
				if (nodeStack.size() <= 0) {
					nodePtr = nullptr;
					return;
				}

				// do one iteration of the dfs sketch above

				DirtyNode* ptr = nodeStack.top();
				nodeStack.pop();

				//if (!discoveredSet.count(ptr)) {

				// if the set has stuff in it OR if it's empty, yield
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
					forwards, depthFirst, true
				);
			} // 200 is out of bounds
		};

		Iterator iterNodes(NodeT* sourceNode, bool forwards = true, bool depthFirst = true, bool includeSelf = true) {
			// build and return an iterator set up in this way
			return Iterator(sourceNode, this, forwards, depthFirst, includeSelf);
		}

		void nodePropagateDirty(int opIndex);


		void nodeInputsChanged(int opIndex) {
			graphChanged = true;
		}
		int checkLegalInput(int opIndex, int testDriverOpIndex) {
			/* if result is 0 all good
			// if not, error and it's illegal
			check for feedback loops
			*/
			if (opIndex == testDriverOpIndex) { // can't connect to itself
				return 1;
			}
			std::vector<int> futureOps = nodesInFuture(opIndex);
			if (std::find(futureOps.begin(), futureOps.end(), testDriverOpIndex) != futureOps.end()) {
				// node found in future, illegal feedback loop
				return 2;
			}
			return 0;
		};


		struct NodeDelta {
			/* given base graph and target graph,
			get nodes to add and nodes to remove, in order to take base
			to target
			WE DON'T DO ANYTHING FANCY TO CHECK TYPES
			*/
			std::vector<std::string> nodesToAdd;
			std::vector<std::string> nodesToRemove;

		};

		struct EdgeDelta {
			/* given base graph and target graph,
			get edges to add and remove, in order to take base
			to target.

			doesn't check nodes - if a node name isn't found in both graphs,
			skip that edge
			*/

			std::map<std::string, std::vector<std::string>> edgesToAdd;
			//std::vector < std::pair<std::string, std::vector<std::string>>> edgesToRemove;

		};

		//NodeDelta getNodeDelta(const DirtyGraph& other,
		//	std::vector<StrataName>& otherNamesToCheck) const {
		//	/* works entirely on names
		//	pass in list of names to consider in other graph
		//	*/
		//	NodeDelta result;
		//	std::vector<StrataName> thisSortedKeys = mapKeys(nameIndexMap);

		//	std::sort(thisSortedKeys.begin(), thisSortedKeys.end());
		//	std::sort(otherNamesToCheck.begin(), otherNamesToCheck.end());
		//	std::set_difference(
		//		otherNamesToCheck.begin(), otherNamesToCheck.end(),
		//		thisSortedKeys.begin(), thisSortedKeys.end(),
		//		std::inserter(result.nodesToAdd, result.nodesToAdd.begin())
		//	);
		//	std::set_difference(
		//		thisSortedKeys.begin(), thisSortedKeys.end(),
		//		otherNamesToCheck.begin(), otherNamesToCheck.end(),
		//		std::inserter(result.nodesToRemove, result.nodesToRemove.begin())
		//	);
		//	return result;
		//}

		//NodeDelta getNodeDelta(const DirtyGraph& other) const {
		//	// get delta of all nodes in graph if not specified
		//	std::vector<StrataName> otherSortedKeys = mapKeys(other.nameIndexMap);
		//	return getNodeDelta(other, otherSortedKeys);
		//}

		//EdgeDelta getEdgeDelta(
		//	const DirtyGraph& other,
		//	//std::vector<std::string>& nodesToCheck 
		//	std::vector<std::string>& nodesToCheck
		//) const {
		//	/* not quite as sleek as nodes
		//	ordering of edges???
		//	probably need a more atomic delta system here

		//	TODO: maybe come back to this, but it probably isn't so necessary -
		//	because of the rules of Strata (eg no going back and changing graph
		//	structure once it's made)
		//	we probably don't need to get too fancy? since edges can only ever be added

		//	for now we just copy the named input map from the target other.
		//	it's too complicated otherwise
		//	*/
		//	EdgeDelta result;
		//	for (auto& nodeNameToCheck : nodesToCheck) {
		//		DirtyNode* thisNode = getNode(nodeNameToCheck);
		//		if (!thisNode) {
		//			continue;
		//		}
		//		auto& name = thisNode->name;
		//		// skip if node from this graph not found in other
		//		if (other.nameIndexMap.find(name) == other.nameIndexMap.end()) {
		//			continue;
		//		}
		//		DirtyNode* otherNode = other.getNode(name);

		//		result.edgesToAdd.insert(std::make_pair(name, otherNode->inputNames()));


		//		//auto thisInputNodes = thisNode->inputNodes();
		//		//int localIndex = 0;
		//		//for (auto otherInputNode : otherNode->inputNodes()) {
		//		//	// skip if this driver node's name in the other graph is not found in this one
		//		//	if (nameIndexMap.find(otherInputNode->name) == nameIndexMap.end()) {
		//		//		continue;
		//		//	}
		//		//	if (thisInputNodes[localIndex]->name != otherInputNode->name) {
		//		//		//result.edgesToAdd.
		//		//	}
		//		//		localIndex += 1;	
		//		//}
		//	}
		//	return result;
		//}

		//virtual DirtyNode* importNode(const DirtyGraph& other, int& thatIndex) {
		//	/* merge in a new node
		//	DOES NOT CHECK EDGES / CONNECTIONS*/
		//	LOG("DirtyGraph import node");
		//	nodes.push_back(other.nodes[thatIndex]);
		//	nodes.back().index = static_cast<int>(nodes.size()) - 1;
		//	nodes.back().graphPtr = this;
		//	l("dirtyGraph imported node is in this graph: " + str(nodes.back().graphPtr == this));
		//	nameIndexMap.insert({ std::string(nodes.back().name), static_cast<int>(nodes.size()) - 1 });
		//	return &nodes.back();
		//}

		//void nodePropagateDirty(int opIndex) {
		//	/* manually set dirty flags on node before calling this function -
		//	all flags set dirty will be propagated to nodes in future*/
		//	// propagate dirty stuff forwards to all nodes
		//	// we add bools to the base state - 
		//	// can't use false arguments to set clean with this function
		//	//LOG("node propagate dirty: " + strata::str(opIndex));
		//	DirtyNode* seedNode = getNode(opIndex);

		//	// if all flags match in dirty map, we consider this node has been visited, and skip tree
		//	Iterator it = iterNodes(
		//		seedNode,
		//		true, // forwards 
		//		true, // depth first
		//		false // include self
		//	);
		//	//for (auto nextNode : it) {
		//	while (it != it.end()) {
		//		if (*it == nullptr) {
		//			//l("iterator reached end null");
		//			break;
		//		}
		//		DirtyNode* nextNode = *it;
		//		if (nextNode->index == seedNode->index) { // skip the same node
		//			it++;
		//			continue;
		//		}
		//		nextNode->dirty = true;
		//		////l("check node: " + nextNode->name);
		//		//bool allMatch = true;
		//		//for (auto pair : seedNode->dirtyMap) {
		//		//	if (!(nextNode->dirtyMap.count(pair.first))) { // propagate if not found
		//		//		allMatch = false;
		//		//		nextNode->dirtyMap[pair.first] = pair.second;
		//		//		continue;
		//		//	}
		//		//	if (nextNode->dirtyMap[pair.first] != pair.second) {
		//		//		allMatch = false;
		//		//		nextNode->dirtyMap[pair.first] = (nextNode->dirtyMap[pair.first] || pair.second); // if either is dirty, keep that value
		//		//	}
		//		//}
		//		//if (allMatch) { // all dirty flags match on this node, skip this tree
		//		//	it.skipTree();
		//		//}
		//		it++;
		//	}
		//}

		//int mergeOther(const DirtyGraph& other, Status& s) {
		//	/* copy nodes from other not found in this graph -
		//	works by NAME ONLY
		//	indices of merged nodes WILL BE DIFFERENT
		//	we also pull in any edge connections from them

		//	LATER add in some way to check / add from only nodes in the critical path of the output
		//	(but with this system, that's guaranteed to be all nodes in the graph?)

		//	returns FIRST INDEX of added nodes, or -1 if none have been added
		//	*/
		//	//LOG("dirtyGraph mergeOther");
		//	int resultIndex = -1;
		//	NodeDelta nDelta = getNodeDelta(other);
		//	// copy nodes to add (required for correct edge delta
		//	for (std::string nodeName : nDelta.nodesToAdd) {
		//		//l("adding other node:" + nodeName);
		//		if (nodeName.empty()) {
		//			//l("other had an empty node name, exiting");
		//			return -1;
		//		}
		//		//importNode(other, other.nameIndexMap.find(nodeName))
		//		int otherIndex = other.nameIndexMap.at(nodeName);
		//		DirtyNode* importedNode = importNode(other, otherIndex);
		//		//l("dirtyGraph imported node is in this graph: " + str(importedNode->graphPtr == this));
		//		if (resultIndex == -1) {
		//			resultIndex = importedNode->index;
		//		}
		//		/* check that imported node has a name*/
		//		if (importedNode->name.empty()) {
		//			//l("imported node name is empty, exiting");
		//			return -1;
		//		}
		//	}
		//	EdgeDelta eDelta = getEdgeDelta(other, nDelta.nodesToAdd);

		//	// node inputs need to be re-indexed based on names
		//	for (auto& nodeInputsPair : eDelta.edgesToAdd) {
		//		DirtyNode* thisNode = getNode(nodeInputsPair.first);
		//		thisNode->inputs.clear();
		//		thisNode->inputs.reserve(eDelta.edgesToAdd[thisNode->name].size());
		//		for (auto inputNodeName : eDelta.edgesToAdd[thisNode->name]) {
		//			thisNode->inputs.push_back(nameIndexMap[inputNodeName]);
		//		}
		//		nodeInputsChanged(thisNode->index);

		//	}
		//	graphChanged = true;
		//	return resultIndex;
		//}

	};

	struct DirtyGraph : DirtyGraphBase<DirtyGraph, DirtyNode> {
	};


}
	//static DirtyGraph testGraph;

	//static auto newNode = testGraph.addNode<>("newNode");













		//};

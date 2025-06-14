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

#include "iterator.h"

/* abstracting pure topo structure of dependency graph - 
assuming that all nodes have a global unique name
(but disregard this if not useful)

we assume a single node is only valid for and belongs to a single type of graph?
ignoring values / evaluation for now


we can't avoid value templating for a node included in an evaluation graph

*/

namespace ed {

	/*template <typename T>
	inline std::set<T> setIntersect(
		std::set<T>& b,
		std::set<T>& b,
		)*/

	//template<typename VT = int>
	struct DirtyGraph;
	//struct EvalGraph<>;

	// override flag type in node type to control how many dirty flags?


	struct DirtyNode {
		int index;
		std::string name;
		DirtyGraph* graphPtr = nullptr;

		using graphT = DirtyGraph;
		inline graphT* getGraphPtr() { return static_cast<graphT*>(graphPtr); }

		bool enabled = true; /* should this be per-node, or a set on the parent object?*/

		/*template<typename GraphT=DirtyGraph>
		virtual GraphT* getGraphPtr() { return ; }*/

		//virtual DirtyGraph* getGraphPtr() { return graphPtr; }

		std::vector<int> inputs; // manage connections yourself

		// my python is showing - we use a map here for easier extensibility
		std::map<const std::string, bool> dirtyMap = { {"main" , true} }; // this is probably pointless
		// TODO: get rid of the map once everything else works

		inline bool setDirty(bool state) {
			/* returns the previously set dirty state*/
			bool oldState = dirtyMap["main"];
			dirtyMap["main"] = state;
			return oldState;
		}

		inline bool anyDirty() {
			for (auto p : dirtyMap) {
				if (p.second) { return true; }
			}
			return false;
		}

		inline std::map<std::string, DirtyNode*> nameInputNodeMap();

		std::vector<DirtyNode*> inputNodes();

		std::vector<std::string> inputNames();

		int temp_inDegree = 0;
		int temp_generation = 0;

		DirtyNode() {}
		DirtyNode(int index, std::string name) : index(index), name(name) {}
		virtual ~DirtyNode(){}

		auto clone() const { return std::unique_ptr<DirtyNode>(clone_impl()); }
		virtual DirtyNode* clone_impl() const { return new DirtyNode(*this); };

		virtual Status postConstructor() {
			/* called after node added to graph, all connections set up*/
			return Status();
		}

		inline void nodeError(std::string errorMsg);
	};

	struct DirtyGraph {
		/* abstracted graph behaviour for topological generations,
		dirty propagation etc
		losing efficiency with indices and names but it's fine

		most things here just stored in arrays

		can we somehow separate the evaluation behaviour from the raw topological graph?

		*/


		/* MEMBERS TO COPY / SERIALISE */
		std::vector<std::unique_ptr<DirtyNode>> nodes;
		std::unordered_map<std::string, int> nameIndexMap;
		int outputIndex = -1; // houdini-esque tracking which node is 
		// designated as the output of the graph

		
		/* TRANSIENT MEMBERS */
		bool graphChanged = true; // if true, needs full rebuild of topo generations
		// topo maps
		std::vector<std::unordered_set<int>> generations;
		// transient map for topology
		std::unordered_map<int, std::unordered_set<int>> nodeDirectDependentsMap;
		std::unordered_map<int, std::unordered_set<int>> nodeAllDependentsMap;

		// map to track nodes that have errored
		std::map<int, std::string> indexErrorMap;

		inline void setOutputNode(int index) {
			outputIndex = index;
			graphChanged = true;
		}

		///////////////// COPYING ////////
		//rule of five (apparently?)
		inline void copyOtherNodesVector(const DirtyGraph& other) {
			// function to deep-copy all nodes in the given vector from the argument graph
			nodes.clear();
			nodes.reserve(other.nodes.size());
			for (auto& ptr : other.nodes) {
				nodes.push_back(std::unique_ptr<DirtyNode>(ptr.get()->clone()));
			}
		}
		virtual void copyOther(const DirtyGraph& other) {
			copyOtherNodesVector(other);
			nameIndexMap = nameIndexMap;
			outputIndex = other.outputIndex;
		}

		DirtyGraph() {}
		~DirtyGraph() = default;
		DirtyGraph(DirtyGraph const& other) {
			copyOther(other);
		}
		DirtyGraph(DirtyGraph&& other) = default;
		DirtyGraph& operator=(DirtyGraph const& other) {
			copyOther(other);
		}
		DirtyGraph& operator=(DirtyGraph&& other) = default;

		auto clone() const { return std::unique_ptr<DirtyGraph>(clone_impl()); }
		template <typename T>
		auto clone() const { return std::unique_ptr<T>(static_cast<T*>(clone_impl())); }
		auto cloneShared() const { return std::shared_ptr<DirtyGraph>(clone_impl()); }
		template <typename T>
		auto cloneShared() const { return std::shared_ptr<T>(static_cast<T*>(clone_impl())); }
		virtual DirtyGraph* clone_impl() const { 
			auto newPtr = new DirtyGraph(*this);
			newPtr->copyOther(*this);
			return newPtr;
		};

		//virtual Status postConstructor() {
		//	/* called after node added to graph, all connections set up*/
		//	return Status();
		//}

		Status setNodeName(DirtyNode* nodePtr, std::string newName) {
			// the one concession to mutable state for now in graphs
			Status s;
			int index = nameIndexMap[nodePtr->name];
			nameIndexMap.erase(nodePtr->name);
			nameIndexMap[newName] = index;
			nodePtr->name = newName;
			return s;
		}


		template <typename NodeT = DirtyNode>
		NodeT* addNode(const std::string& name, bool _callPostConstructor=true) {
			/* create new node object,
			returns a new pointer to it.
			
			callPostConstructor true by default - 
			if a derived graph type calls this, set to false to call post
			only at the end of the overridden function

			should we convert all this to return status code?
			*/
			if (nameIndexMap.count(name) != 0) {
				Status s;
				CWMSG(s, "Name " + name + " already found in dirty graph, returning nullptr");
				return nullptr;
			}

			const int newIndex = static_cast<int>(nodes.size());
			auto nodePtr = std::make_unique<NodeT>(newIndex, name);
			nodes.push_back(std::move(nodePtr));

			NodeT* newNodePtr = static_cast<NodeT*>(nodes[newIndex].get());
			nameIndexMap[newNodePtr->name] = newIndex;
			newNodePtr->graphPtr = this;
			graphChanged = true;

			if (_callPostConstructor) {
				Status s = newNodePtr->postConstructor();
				CWMSG(s, "post-constructor on node " + newNodePtr->name + " failed!")
				
			}
			return newNodePtr;
		}

		inline DirtyNode* getNode(DirtyNode*& node) const {
			// included here for similar syntax to get op pointer, no matter the input
			// not sure if this is actually useful in c++
			return node;
		}
		inline DirtyNode* getNode(const int& index) const {
			//DEBUGSL("get node by index " + std::to_string(index));
			//DEBUGSL("nodes size: " + std::to_string(nodes.size()));
			if (nodes.size() <= index) {
				return nullptr;
			}
			return nodes[index].get();
		}
		inline DirtyNode* getNode(const std::string& nodeName) const {
			
			auto check = nameIndexMap.find(nodeName);
			if (check == nameIndexMap.end()) { return nullptr; }
			return nodes[check->second].get();
		}

		template<typename NodeT>
		inline NodeT* getNode(DirtyNode*& node) const {
			return dynamic_cast<NodeT*>(getNode(node));
		}
		template<typename NodeT>
		inline NodeT* getNode(const int& index) const {
			return dynamic_cast<NodeT*>(getNode(index));
		}
		template<typename NodeT>
		inline NodeT* getNode(const std::string& nodeName) const {
			return dynamic_cast<NodeT*>(getNode(nodeName));
		}


		template<typename argT>
		inline std::vector<DirtyNode*> getNodes(argT* start, argT* end) {
			/* this won't filter out unknown nodes, you'll just get nullptrs
			in the returned vector. That way size stays the same*/
			std::vector<DirtyNode*> result;
			while (start != end) {
				result.push_back(getNode(*start));
				start++;
			}
			return result;
		}

		inline void addNodeError(int index, std::string& errorMsg) {
			indexErrorMap.insert(std::make_pair(index, errorMsg)
			);
		}

		const inline std::string outputNodeName() {
			if (outputIndex == -1) {
				return "";
			}
			return nodes[outputIndex]->name;
		}

		/*template<typename seqT>
		inline seqT<int> namesToIndices(seqT<str> names) {
		}*/
		// couldn't work out how to template on a template like this

		inline std::vector<int> namesToIndices(std::string* start, std::string* end) {
			std::vector<int> result;
			while (start != end) {
				result.push_back(nameIndexMap[*start]);
				start++;
			}
			return result;
		}
		inline std::vector<std::string> indicesToNames(int* start, int* end) {
			std::vector<std::string> result;
			while (start != end) {
				result.push_back(nodes[*start].get()->name);
				start++;
			}
			return result;
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
			DEBUGS("gather nodes no inputs");
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
			DEBUGS("gathered seed nodes");
			/* build generations
			*
			*/
			while (zeroDegree.size()) {
				DEBUGS("seed outer iter, zdegree:" + std::to_string(zeroDegree.size()) + "result: " + std::to_string(result.size()));
				result.push_back(zeroDegree); // add all zero degree generation to result
				zeroDegree.clear();

				// check over all direct children of current generation
				for (auto& nodeId : result[seqIndex(-1, result.size())]) {
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
			DEBUGS("built generations");
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
			DEBUGS("set generation ids");
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


		inline void syncGraphStructure() {
			if (graphChanged) {
				Status s;
				rebuildGraphStructure(s);
			}
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
			DEBUGSL("nodesInHistory flat " + std::to_string(opIndex));
			std::unordered_set<int> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					DirtyNode* checkOp = getNode(checkNodeIndex);
					newToCheck.insert(checkOp->inputs.begin(), checkOp->inputs.end());

					// add the inputs to result this iteration to
					result.insert(checkNodeIndex);
					//result.insert(checkOp->inputs.begin(), checkOp->inputs.end());
				}
				toCheck = newToCheck;
			}
			DEBUGSL("nodesInHistory result");
			DEBUGVI(result);
			return result;
		}

		std::vector<std::vector<int>> nodesInHistory(int opIndex, bool returnGenerations) {
			/* generation list of nodes in history
			return generations where latest nodes guaranteed to be included before deeper nodes
			
			*/
			std::unordered_map<int, int> visited; // {index : highest degree}
			std::stack<int> toCheck;
			toCheck.push(opIndex);

			visited[opIndex] = 0;

			int maxDepth = 0;
			while (toCheck.size()) {
				int checkIndex = toCheck.top();
				toCheck.pop();
				for (int inIndex : getNode(checkIndex)->inputs) {
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

		//SmallList<std::unordered_set<int>, 8> nodesInFuture(int opIndex, bool returnGenerations) {
		std::vector<std::unordered_set<int>> nodesInFuture(int opIndex, bool returnGenerations) {
			/* generation list of nodes in future*/
			//SmallList<std::unordered_set<int>, 8> result;
			std::vector<std::unordered_set<int>> result;

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
				nodesToCheck.insert(node.get()->index);
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
						nextNode->dirtyMap[pair.first] = (nextNode->dirtyMap[pair.first] || pair.second); // if either is dirty, keep that value
					}
				}
				if (allMatch) { // all dirty flags match on this node, skip this tree
					it.skipTree();
				}
			}
		}


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
			std::unordered_set<int> futureOps = nodesInFuture(opIndex);
			if (futureOps.count(testDriverOpIndex)) {
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
		
		NodeDelta getNodeDelta(const DirtyGraph& other,
			std::vector<std::string>& otherNamesToCheck) const {
			/* works entirely on names
			pass in list of names to consider in other graph
			*/
			NodeDelta result;
			std::vector<std::string> thisSortedKeys = mapKeys(nameIndexMap);
			//std::vector<std::string> otherSortedKeys(key_begin(other.nameIndexMap), key_end(other.nameIndexMap));

			std::sort(thisSortedKeys.begin(), thisSortedKeys.end());
			std::sort(otherNamesToCheck.begin(), otherNamesToCheck.end());
			std::set_difference(
				otherNamesToCheck.begin(), otherNamesToCheck.end(),
				thisSortedKeys.begin(), thisSortedKeys.end(),
				result.nodesToAdd.begin()
			);
			std::set_difference(
				thisSortedKeys.begin(), thisSortedKeys.end(),
				otherNamesToCheck.begin(), otherNamesToCheck.end(),
				result.nodesToRemove.begin()
			);
			return result;
		}

		NodeDelta getNodeDelta(const DirtyGraph& other) const {
			// get delta of all nodes in graph if not specified
			std::vector<std::string> otherSortedKeys = mapKeys(other.nameIndexMap);
			return getNodeDelta(other, otherSortedKeys);
		}

		EdgeDelta getEdgeDelta(
			const DirtyGraph& other,
			//std::vector<std::string>& nodesToCheck 
			std::vector<std::string>& nodesToCheck
			) const {
			/* not quite as sleek as nodes
			ordering of edges???
			probably need a more atomic delta system here

			TODO: maybe come back to this, but it probably isn't so necessary - 
			because of the rules of Strata (eg no going back and changing graph
			structure once it's made)
			we probably don't need to get too fancy? since edges can only ever be added

			for now we just copy the named input map from the target other.
			it's too complicated otherwise
			*/
			EdgeDelta result;
			for (auto& nodeNameToCheck : nodesToCheck) {
				DirtyNode* thisNode = getNode(nodeNameToCheck);
				if (!thisNode) {
					continue;
				}
				auto& name = thisNode->name;
				// skip if node from this graph not found in other
				if (other.nameIndexMap.find(name) == other.nameIndexMap.end()) {
					continue;
				}				
				DirtyNode* otherNode = other.getNode(name);

				result.edgesToAdd.insert(std::make_pair(name, otherNode->inputNames()));
				

				//auto thisInputNodes = thisNode->inputNodes();
				//int localIndex = 0;
				//for (auto otherInputNode : otherNode->inputNodes()) {
				//	// skip if this driver node's name in the other graph is not found in this one
				//	if (nameIndexMap.find(otherInputNode->name) == nameIndexMap.end()) {
				//		continue;
				//	}
				//	if (thisInputNodes[localIndex]->name != otherInputNode->name) {
				//		//result.edgesToAdd.
				//	}
				//		localIndex += 1;	
				//}
			}
			return result;
		}

		virtual DirtyNode* importNode(const DirtyGraph& other, int& thatIndex) {
			/* merge in a new node
			DOES NOT CHECK EDGES / CONNECTIONS*/
			nodes.push_back(other.nodes[thatIndex]->clone());
			nodes.back().get()->index = static_cast<int>(nodes.size()) - 1;
			nameIndexMap[nodes.back().get()->name] = static_cast<int>(nodes.size()) - 1;
			return nodes.back().get();
		}

		virtual int mergeOther(const DirtyGraph& other, Status& s) {
			/* copy nodes from other not found in this graph - 
			works by NAME ONLY
			indices of merged nodes WILL BE DIFFERENT
			we also pull in any edge connections from them

			LATER add in some way to check / add from only nodes in the critical path of the output
			(but with this system, that's guaranteed to be all nodes in the graph?)

			returns FIRST INDEX of added nodes, or -1 if none have been added
			*/
			int resultIndex = -1;
			NodeDelta nDelta = getNodeDelta(other);
			// copy nodes to add (required for correct edge delta
			for (std::string nodeName : nDelta.nodesToAdd) {
				//importNode(other, other.nameIndexMap.find(nodeName))
				int otherIndex = other.nameIndexMap.at(nodeName);
				DirtyNode* importedNode = importNode(other, otherIndex );
				if (resultIndex == -1) {
					resultIndex = importedNode->index;
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



	};

	template <typename VT>
	struct EvalGraph;

	struct EvalAuxData {
		/* aux data struct passed alongside graph evaluation - 
		this is just cheat to get expressions and geometry working
		with the same framework. No guarantees on ordering/threading, beyond
		what the structure of the graph provides.
		Suggestions on the correct way to 
		structure all this are welcome
		*/
	};

	template <typename VT>
	struct EvalNode : DirtyNode {
		
		using DirtyNode::DirtyNode;

		// couldn't work out how to bind instances to functions, so
		// by default eval functions are static methods
		//using EvalFnT = Status(EvalNode<VT>::*)(VT&, Status&);
		using EvalFnT = Status(*)(EvalNode<VT>*, VT&, Status&);
		typedef EvalFnT EvalFnT;

		static Status evalMain(EvalNode<VT>* node, VT& value, Status& s) { return s; }
		EvalFnT evalFnPtr = evalMain; // pointer to op function - if passed, easier than defining custom classes for everything?

		// ordered map to define evaluation order of steps
		std::map<const std::string, EvalFnT&> evalFnMap{
			{"main" , evalFnPtr}
		};
		// all the stuff above is SO complicated, just using one eval function and
		// trusting user to check their own logic behind it?

		// do we actually gain anything from multiple passes?
		// not right now
		// stop it
		// get some help
		// you're tearing me apart lisa


		// for strata, node eval should be valid as a member function or as a 
		// written function in StrataL
		// is there any point in this being static? I have no idea how to get an unknown node to run the right class eval function if it is
		/*static Status eval(EvalNode<VT>* node, VT& value, 
			EvalAuxData* auxData, Status& s) { return s; }*/
		virtual Status eval(VT& value,
			EvalAuxData* auxData, Status& s) {
			return s;
		}

		virtual EvalNode* clone_impl() const { return new EvalNode(*this); };


		virtual Status preEval(const std::string& stepName, Status& s) {
			// called before each step is run
			// use this to recompile parametres, if they've changed
			return s;
		}

		virtual void preReset() {
			// before node value is reset in graph
		}
		virtual void postReset() {
			// after node value is reset in graph
		};
		//EvalGraph<VT>* graphPtr = nullptr;

		virtual EvalGraph<VT>* getGraphPtr() { return reinterpret_cast<EvalGraph<VT>*>(graphPtr); }

		VT& value() { // retrieve whatever node's current value is in graph
			//return &(graphPtr->results[index]);
			return getGraphPtr()->results[index];
		}

	};

	//struct EvalGraph<int>;

	
	struct NodeData  {
		/* TEMP maybe - should be possible to declare custom block of
		common node data for each node, probably template the EvalGraph on
		this too
		for now we just use this type*/
		std::vector<std::unordered_set<int>> elementsAffected; 
	};

	template <typename VT>
	struct EvalGraph : DirtyGraph {
		// adding mechanisms for evaluation and caching results

		VT baseValue; // use to copy and reset node result entries
		std::vector<VT> results;

		// this set should go in the strataGraph subclass, but this is
		// easier for now

		/* NO IDEA how to handle outputs vs islands in graph - this seems way simpler.
		single node to eval up to. don't implicitly merge anything.

		this is SEPARATE to nodes being enabled/disabled, all this affects is the critical path
		in the graph
		*/
		int outNodeIndex = -1;
		
		std::vector<NodeData> nodeDatas; 

		template <typename T>
		auto cloneShared(bool copyAllResults) const { return std::shared_ptr<T>(static_cast<T*>(clone_impl(copyAllResults))); }
		virtual EvalGraph<VT>* clone_impl(bool copyAllResults) const {
			auto newPtr = new EvalGraph<VT>(*this);
			newPtr->copyOther(*this, copyAllResults);
			return newPtr;
		};

		virtual void copyOther(const EvalGraph& other, bool copyAllResults=true) {
			DirtyGraph::copyOther(other);
			nodeDatas = other.nodeDatas;
			if (copyAllResults) {
				results = other.results;
			}
			else { // only copy result of output node
				results.clear();
				results.resize(other.results.size());
				results[outputIndex] = other.results[outputIndex];
			}
		}

		EvalGraph() {
		};
		~EvalGraph() = default;
		EvalGraph(EvalGraph const& other) {
			copyOther(other);
		}
		EvalGraph(EvalGraph&& other) = default;
		EvalGraph& operator=(EvalGraph const& other) {
			copyOther(other);
		}
		EvalGraph& operator=(EvalGraph&& other) = default;


		virtual int mergeOther(const EvalGraph& other, bool mergeAllResults, Status& s) {
			/* if not mergeAllResults, only copy across the result value
			* of the graph's output node, if it's part of the nodes merged
			* 
			* maybe this should be an extension of ImportNode instead
			*/
			int result = DirtyGraph::mergeOther(other, s);
			if (s) {
				return result;
			}
			// if we want to merge all results or the other graph has no output index set,
			// copy everything
			if (mergeAllResults || (other.outputIndex == -1)) {
				for (int i = result; i < static_cast<int>(nodes.size()); i++) {
					int otherIndex = other.nameIndexMap.at(getNode(i)->name);
					results[i] = other.results[otherIndex];
				}
			}
			else { // copy only the value of the result node
				DirtyNode* otherOutNode = other.getNode(outputIndex);
				results[nameIndexMap[otherOutNode->name]] = other.results[otherOutNode->index];
			}
			return result;
		}


		template <typename NodeT = EvalNode<VT>>
		NodeT* addNode(const std::string& name, VT defaultValue = VT(),
			typename NodeT::EvalFnT evalFnPtr = nullptr) 
		{
			NodeT* baseResult = DirtyGraph::addNode<NodeT>(name, false);
			//if (defaultValue == nullptr) { // if default is given, add it to graph
			//	defaultValue = VT();
			//}
			results.push_back(defaultValue);
			nodeDatas.push_back(NodeData());
			baseResult->postConstructor();

			//if (evalFnPtr != nullptr) {// if evalFn is given, set it on node
			//	baseResult->evalFnPtr = evalFnPtr;
			//}
			return baseResult;
		}

		void clear() {
			nodes.clear();
			results.clear();
			nodeDatas.clear();
			graphChanged = true;
		}

		//template <typename AuxT=int>
		Status evalNode(EvalNode<VT>* op, EvalAuxData* auxData, Status& s) {
			/* encapsulated evaluation for a single op
			* we guarantee that all an op's input nodes will already
			* have been evaluated.
			* 
			* is it worth adding capability for multiple passes? multiple operations per pass?
			* 
			* flag if a certain node requires a certain set of passes to run in the graph up until that point?
			* 
			we're just gonna use a single eval function and rely on 
			* user code to undertake the herculaean task
			* of writing if statements
			* to only evaluate the bits of the nodes they need
			*/
			DEBUGSL("evalNode begin");
			// reset node state
			op->preReset();
			// copy base geo into node's result, if it has inputs
			if (op->inputs.size()) {
				results[op->index] = results[op->inputs[0]];
			}
			else { // if no inputs, copy graph's baseValue
				results[op->index] = baseValue;
			}
			op->postReset();

			/*TODO: check for errors in evaluation, somehow -
			that would mean passing out errors to graph eval function

			...we could hypothetically CHECK an MSTATUS...
			...and hypothetically RETURN_IT...
			*/


			/* for each evalFn, check if we find a dirty flag for its key,
			* and if dirty, evaluate it
			* 
			* we also set that flag clean automatically
			*/
			//for (auto fnPair : op->evalFnMap) {

			op->preEval("main", s);
			CWRSTAT(s, "error in preEval for node: " + op->name);
			DEBUGS("pre eval nResults, " + std::to_string(results.size()));
			op->eval( results[op->index], auxData, s); // illegal read on this line apparently
			CWRSTAT(s, "error in EVAL for node: " + op->name);

			// donezo
			return s;
		}

		//template< typename AuxT=int>
		Status evalGraph(Status& s, int upToNodeId = -1, EvalAuxData* auxData=nullptr) {
			// TODO: don't eval topo and data all the time obviously
			// just for now

			/* first input of each node by default should be the manifold stream
			to write on. if a node has no input, it's an ancestor in the graph-
			create an empty manifold object for that object

			I don't know if there's an alternative to copying the manifold to every node's output?
			surely that's ridiculously wasteful.
			but using a constant reference means we can't have a "breakpoint" in the graph, since the same
			object will be edited constantly?

			until a node branches, we can work on only one object?

			copy everything for now. make it exist, then make it efficient.

			*/

			// sort topo generations in nodes if graph has changed
			DEBUGSL("EvalGraph begin eval to node: " + std::to_string(upToNodeId));
			//return s;
			if (graphChanged) {
				DEBUGS("rebuilding graph structure")
				rebuildGraphStructure(s);
				DEBUGS("structure rebuild complete")
			}
			CWRSTAT(s, "ERROR rebuilding graph structure ahead of graph eval, halting");

			// if specific node given, 
			std::unordered_set<int> toEval;
			if (upToNodeId > -1) {
				toEval = nodesInHistory(upToNodeId);
			}
			else { // run everything
				toEval.reserve(nodes.size());
				for (int n = 0; n < nodes.size(); n++) { // does c++ have a linear space
					toEval.insert(n);
				}
			}
			DEBUGSL("toEval: ");
			DEBUGVI(toEval);
			DEBUGS("nGenerations " + std::to_string(generations.size()));

			bool foundEndNode = false;
			// for now just work in generations
			// go through generations in sequence
			for (auto generation : generations) {
				DEBUGSL("generation");
				DEBUGVI(generation);
				// go through each node in generation
				// this is the bit that can be parallelised
				if (upToNodeId > -1) {
					DEBUGS("doing intersection")
					std::unordered_set<int> baseGeneration(generation);
					generation.clear();
					std::set_intersection(
						baseGeneration.begin(), baseGeneration.end(),
						toEval.begin(), toEval.end(),
						std::inserter(generation, generation.begin())
					); // god c++ is so complicated
					DEBUGS("generation after intersection");
					DEBUGVI(generation);
				}
				
				
				for (auto& nodeId : generation) {
					EvalNode<VT>* node = static_cast<EvalNode<VT>*>(getNode(nodeId));
					// check if anything on node is dirty - if so, continue
					if (!node->anyDirty()) {
						continue;
					}
					// eval whole node
					evalNode(node, auxData, s);
					CRMSG(s, "ERROR eval-ing op " + node->name + ", halting Strata graph ");

					// if node's index is given as breakpoint to eval to, end
					if (node->index == upToNodeId) {
						foundEndNode = true;
						break;
					}
				}
				// stop all eval if end node reached
				if (foundEndNode) {
					break;
				}
			}
			return s;
		}

		virtual DirtyNode* importNode(const EvalGraph& other, int& thatIndex) {
			/* merge in a new node, and add entry in results for it?*/
			nodes.push_back(other.nodes[thatIndex]->clone());
			nodes.back().get()->index = static_cast<int>(nodes.size()) - 1;
			nameIndexMap[nodes.back().get()->name] = static_cast<int>(nodes.size()) - 1;
			return nodes.back().get();
		}

	};


}
	//static DirtyGraph testGraph;

	//static auto newNode = testGraph.addNode<>("newNode");













		//};

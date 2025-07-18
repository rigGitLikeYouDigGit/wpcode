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


	//struct DirtyNode : public StaticClonable<DirtyNode> {
	struct DirtyNode  {
		int index = -1;
		std::string name = "_";
		DirtyGraph* graphPtr = nullptr;

		using graphT = DirtyGraph;
		inline graphT* getGraphPtr() { return static_cast<graphT*>(graphPtr); }

		bool enabled = true; /* should this be per-node, or a set on the parent object?*/

		/*template<typename GraphT=DirtyGraph>
		virtual GraphT* getGraphPtr() { return ; }*/

		//virtual DirtyGraph* getGraphPtr() { return graphPtr; }

		std::vector<int> inputs = {}; // manage connections yourself

		int temp_inDegree = 0;
		int temp_generation = 0;

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


		DirtyNode() {}
		DirtyNode(int index, std::string name) : index(index), name(name) {}
		virtual ~DirtyNode(){}

		virtual std::unique_ptr<DirtyNode> clone() const { return std::unique_ptr<DirtyNode>(this->clone_impl()); }
		template<typename T>
		std::unique_ptr<T> clone() { 
			//return std::unique_ptr<T>(dynamic_cast<T*>(this->clone_impl()));
			return std::unique_ptr<T>(this->clone_impl<T>());
		}
		virtual DirtyNode* clone_impl() const { return new DirtyNode(*this); };

		template <typename T>
		T* clone_impl() const { return new T(*reinterpret_cast<const T*>(this)); }



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

		///////////////// COPYING ////////
		//rule of five (apparently?)
		inline void copyOtherNodesVector(const DirtyGraph& other) {
			// function to deep-copy all nodes in the given vector from the argument graph
			LOG("COPY OTHER NODES dirtyGraph");
			nodes.clear();
			nodes.reserve(other.nodes.size());
			for (auto& ptr : other.nodes) {
				//nodes.push_back(std::unique_ptr<DirtyNode>(ptr.get()->clone()));
				nodes.push_back(ptr.get()->clone<DirtyNode>());
			}
			for (auto& ptr : nodes) {
				ptr->graphPtr = this;
			}
		}

		virtual void copyOther(const DirtyGraph& other) {
			LOG("GRAPH COPY OTHER, other out index:" + str(other.getOutputIndex()));
			this->copyOtherNodesVector(other);
			nameIndexMap = other.nameIndexMap;
			_outputIndex = other._outputIndex;
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
			//nameIndexMap[newName] = index;
			nameIndexMap.insert({ std::string(newName), index });
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
			//if (nameIndexMap.count(name) != 0) {
			if (nameIndexMap.find(name) != nameIndexMap.end()) {
				Status s;
				CWMSG(s, "Name " + name + " already found in dirty graph, returning nullptr");
				return nullptr;
			}

			const int newIndex = static_cast<int>(nodes.size());
			auto nodePtr = std::make_unique<NodeT>(newIndex, name);
			nodes.push_back(std::move(nodePtr));

			NodeT* newNodePtr = static_cast<NodeT*>(nodes[newIndex].get());
			//nameIndexMap[std::string(newNodePtr->name)] = newIndex;
			nameIndexMap.insert({ std::string(newNodePtr->name), newIndex });
			
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
			return nodes[getOutputIndex()]->name;
		}

		inline bool hasOutputNode() {
			return (nodes.size() && (_outputIndex > -1));
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
				l("check node " + node->name + " inputs: ");
				DEBUGVI(node->inputs);
				node->temp_inDegree = static_cast<int>(node->inputs.size());
				// build topology outputs
				int inputCount = 0;
				for (int inputId : node->inputs) {
					if (inputId < 0) {
						continue;
					}
					// add this node to the dependents of all its inputs
					nodeDirectDependentsMap[inputId].push_back(node->index);
					inputCount += 1;
				}
				// if node has no inputs, put it in first generation
				if (!inputCount) {
					zeroDegree.push_back(node->index);
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
				std::string nodeName(nodes[i]->name);
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

			std::stack<DirtyNode*> nodeStack = {};
			std::unordered_set<DirtyNode*> discoveredSet = {};

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

		Iterator iterNodes(DirtyNode* sourceNode, bool forwards = true, bool depthFirst = true, bool includeSelf = true) {
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
		
		NodeDelta getNodeDelta(const DirtyGraph& other,
			std::vector<std::string>& otherNamesToCheck) const {
			/* works entirely on names
			pass in list of names to consider in other graph
			*/
			NodeDelta result;
			std::vector<std::string> thisSortedKeys = mapKeys(nameIndexMap);

			std::sort(thisSortedKeys.begin(), thisSortedKeys.end());
			std::sort(otherNamesToCheck.begin(), otherNamesToCheck.end());
			std::set_difference(
				otherNamesToCheck.begin(), otherNamesToCheck.end(),
				thisSortedKeys.begin(), thisSortedKeys.end(),
				std::inserter(result.nodesToAdd, result.nodesToAdd.begin())
			); 
			std::set_difference(
				thisSortedKeys.begin(), thisSortedKeys.end(),
				otherNamesToCheck.begin(), otherNamesToCheck.end(),
				std::inserter(result.nodesToRemove, result.nodesToRemove.begin())
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
			LOG("DirtyGraph import node");
			nodes.push_back(other.nodes[thatIndex]->clone());
			nodes.back().get()->index = static_cast<int>(nodes.size()) - 1;
			nodes.back()->graphPtr = this;
			l("dirtyGraph imported node is in this graph: " + str(nodes.back()->graphPtr == this));
			nameIndexMap.insert({std::string( nodes.back().get()->name), static_cast<int>(nodes.size()) - 1 });
			return nodes.back().get();
		}

		virtual int mergeOther(const DirtyGraph& other, Status& s);


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

		consider this representing 
		*/

		virtual Status& mergeOther(Status& s, EvalAuxData* other) {
			/* this is used after eval'ing nodes in parallel, to combine mutated
			auxDatas from those nodes, and pass new result to next generation
			maybe it should be
			changed to take a vector*/
			return s;
		}
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
			LOG("EVALNODE EVAL on node " + name + " - THIS IS WRONG");
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
		std::vector<VT> results = {};

		// this set should go in the strataGraph subclass, but this is
		// easier for now

		/* NO IDEA how to handle outputs vs islands in graph - this seems way simpler.
		single node to eval up to. don't implicitly merge anything.

		this is SEPARATE to nodes being enabled/disabled, all this affects is the critical path
		in the graph
		*/
		
		//std::vector<NodeData> nodeDatas = {};

		template <typename T>
		auto cloneShared(bool copyAllResults) const {
			LOG("EvalGraph cloneShared");
			return std::shared_ptr<T>(
				static_cast<T*>(T::clone_impl(copyAllResults))); }
		//auto cloneShared(bool copyAllResults) const { return std::shared_ptr<T>(dynamic_cast<T*>(clone_impl(copyAllResults))); }
		virtual EvalGraph<VT>* clone_impl(bool copyAllResults) const {
			LOG("EvalGraph clone impl");
			auto newPtr = new EvalGraph<VT>(*this);
			//newPtr->copyOther(*this, copyAllResults);
			return newPtr;
		};

		virtual void copyOther(const EvalGraph& other, bool copyAllResults=true) {
			LOG("EVAL GRAPH COPY OTHER, other nodes: " + str(other.nodes.size()))
			DirtyGraph::copyOther(other);
			//nodeDatas = other.nodeDatas;
			/* if graph is empty, it doesn't matter*/
			if (!nodes.size()) {
				return;
			}
			if (copyAllResults) {
				results = other.results;
			}
			else { // only copy result of output node
				results.clear();
				results.resize(other.results.size());
				results[getOutputIndex()] = other.results[getOutputIndex()];
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
			return *this;
		}
		EvalGraph& operator=(EvalGraph&& other) = default;


		virtual int mergeOther(const EvalGraph& other, bool mergeAllResults, Status& s) {
			/* if not mergeAllResults, only copy across the result value
			* of the graph's output node, if it's part of the nodes merged
			* 
			* maybe this should be an extension of ImportNode instead
			*/
			LOG("EVAL GRAPH merge other, other n nodes: " + str(other.nodes.size()));
			int result = DirtyGraph::mergeOther(other, s);
			if (s) {
				l("error in dirtyGraph merge, exiting ");
				return result;
			}
			if (!other.nodes.size()) {
				l("other graph has no nodes, exiting");
				return result;
			}
			results.resize(std::max(nodes.size(), other.nodes.size()));

			// if we want to merge all results or the other graph has no output index set,
			// copy everything
			if (mergeAllResults || (other._outputIndex == -1)) {
				l("merging all values");
				// start iteration at first node added from other graph
				for (int i = result; i < static_cast<int>(nodes.size()); i++) {
					int otherIndex = other.nameIndexMap.at(getNode(i)->name);
					results[i] = other.results[otherIndex];
				}
			}
			else { // copy only the value of the result node
				
				DirtyNode* otherOutNode = other.getNode(other._outputIndex);
				l("merge single result value from node: " + otherOutNode->name + " " + str(otherOutNode->index));
				results[nameIndexMap[otherOutNode->name]] = other.results[other._outputIndex];
			}
			return result;
		}


		template <typename NodeT = EvalNode<VT>>
		NodeT* addNode(const std::string& name, VT defaultValue = VT(),
			typename NodeT::EvalFnT evalFnPtr = nullptr) 
		{
			NodeT* baseResult = DirtyGraph::addNode<NodeT>(name, false);
			if (baseResult == nullptr) {
				LOG("baseResult null, dupe node found")
				return nullptr; };
			//if (defaultValue == nullptr) { // if default is given, add it to graph
			//	defaultValue = VT();
			//}
			results.push_back(defaultValue);
			//nodeDatas.push_back(NodeData());
			baseResult->postConstructor();

			//if (evalFnPtr != nullptr) {// if evalFn is given, set it on node
			//	baseResult->evalFnPtr = evalFnPtr;
			//}
			return baseResult;
		}

		void clear() {
			nodes.clear();
			results.clear();
			//nodeDatas.clear();
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
			LOG("evalNode begin: " + op->name + " " + str(op->index) + " n results: " + str(results.size()) );
			l("op inputs:");
				DEBUGVI(op->inputs);
			// reset node state
			op->preReset();
			// copy base geo into node's result, if it has inputs
			int resultSet = 0;
			if (op->inputs.size()) {
				if (op->inputs[0] > -1) {
					l("copying main input from index:" + str(op->inputs[0]));
					results[op->index] = results[op->inputs[0]];
					//l("after copying result:" + results[op->index].printInfo());
					resultSet = 1;
				}				
			}
			if(!resultSet) { // if no inputs, copy graph's baseValue
				l("inputs not set, copying base value");
				results[op->index] = baseValue;
			}
			op->postReset();


			op->preEval("main", s);
			CWRSTAT(s, "error in preEval for node: " + op->name);
			l("pre eval nResults, " + std::to_string(results.size()));
			s = op->eval(results[op->index], auxData, s);
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
			LOG("EvalGraph begin eval to node: " + std::to_string(upToNodeId));
			//return s;
			if (graphChanged) {
				l("rebuilding graph structure");
				rebuildGraphStructure(s);
				l("structure rebuild complete");
			}
			CWRSTAT(s, "ERROR rebuilding graph structure ahead of graph eval, halting");

			// if specific node given, 
			std::vector<int> toEval;
			if (upToNodeId > -1) {
				toEval = nodesInHistory(upToNodeId);
			}
			else { // run everything
				toEval.reserve(nodes.size());
				for (int n = 0; n < nodes.size(); n++) { // does c++ have a linear space
					toEval.push_back(n);
				}
			}
			l("toEval: ");
			DEBUGVI(toEval);
			l("nGenerations " + std::to_string(generations.size()));

			bool foundEndNode = false;
			// for now just work in generations
			// go through generations in sequence
			for (auto generation : generations) {
				l("generation");
				DEBUGVI(generation);
				// go through each node in generation
				// this is the bit that can be parallelised
				if (upToNodeId > -1) {
					l("doing intersection");
					std::vector<int> baseGeneration(generation);
					generation.clear();
					std::sort(baseGeneration.begin(), baseGeneration.end());
					std::sort(toEval.begin(), toEval.end());
					std::set_intersection(
						baseGeneration.begin(), baseGeneration.end(),
						toEval.begin(), toEval.end(),
						std::inserter(generation, generation.begin())
					); // god c++ is so complicated
					l("generation after intersection");
					DEBUGVI(generation);
				}

				if (!generation.size()) {
					l("generation empty, skipping");
					continue;
				}
				
				
				for (auto& nodeId : generation) {
					EvalNode<VT>* node = static_cast<EvalNode<VT>*>(getNode(nodeId));
					l("consider node: " + str(node->name) + ", " + str(node->index));
					// check if anything on node is dirty - if so, continue
					if (!node->anyDirty()) {
						l("node is not dirty, skip");
						continue;
					}
					// eval whole node
					evalNode(node, auxData, s);
					CRMSG(s, "ERROR eval-ing op " + node->name + ", halting Strata graph ");

					// set node clean
					node->setDirty(false);

					// if node's index is given as breakpoint to eval to, end
					if (node->index == upToNodeId) {
						foundEndNode = true;
						break;
					}
				}
				// stop all eval if end node reached
				if (foundEndNode) {
					l("found end node, ending eval");
					break;
				}
			}
			l("end graph eval");
			return s;
		}

		Status evalGraphSerial(Status& s, int upToNodeId = -1, EvalAuxData* auxData = nullptr) {
			/*eval without fancy parallel generations*/
			LOG("EvalGraphSerial begin eval to node: " + std::to_string(upToNodeId));

			// if specific node given, 
			std::vector<int> toEval;
			if (upToNodeId > -1) {
				toEval = nodesInHistory(upToNodeId);
			}
			else { // run everything
				//toEval.reserve(nodes.size());
				//for (int n = 0; n < nodes.size(); n++) { // does c++ have a linear space
				//	toEval.push_back(n);
				//}
				toEval = nodesInHistory(getOutputIndex());
			}
			
			for (size_t i = 0; i < toEval.size(); i++) {
				int nodeId = toEval.rbegin()[i];
				EvalNode<VT>* node = static_cast<EvalNode<VT>*>(getNode(nodeId));
				if (!node->anyDirty()) {
					l("node is not dirty, skip");
					continue;
				}
				// eval whole node
				evalNode(node, auxData, s);
				CRMSG(s, "ERROR eval-ing op " + node->name + ", halting Strata graph ");

				// set node clean
				node->setDirty(false);

			}
			l("end serial graph eval");
			return s;
		}

		virtual DirtyNode* importNode(const EvalGraph& other, int& thatIndex) {
			/* merge in a new node, and add entry in results for it?*/
			LOG("EvalGraph import node");
			nodes.push_back(other.nodes[thatIndex]->clone());
			nodes.back().get()->index = static_cast<int>(nodes.size()) - 1;
			nodes.back()->graphPtr = this;
			l("evalGraph imported node is in this graph: " + str(nodes.back()->graphPtr == this));

			nameIndexMap.insert({ std::string(nodes.back().get()->name), static_cast<int>(nodes.size()) - 1 });
			return nodes.back().get();
		}

	};


}
	//static DirtyGraph testGraph;

	//static auto newNode = testGraph.addNode<>("newNode");













		//};

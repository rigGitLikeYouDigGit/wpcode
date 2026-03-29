
#include "./dirtyGraph.h"


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

	//// couldn't work out how to bind instances to functions, so
	//// by default eval functions are static methods
	////using EvalFnT = Status(EvalNode<VT>::*)(VT&, Status&);
	//using EvalFnT = Status(*)(EvalNode<VT>*, VT&, Status&);
	//typedef EvalFnT EvalFnT;

	static Status evalMain(EvalNode<VT>* node, VT& value, Status& s) { return s; }
	//EvalFnT evalFnPtr = evalMain; // pointer to op function - if passed, easier than defining custom classes for everything?

	//// ordered map to define evaluation order of steps
	//std::map<const std::string, EvalFnT&> evalFnMap{
	//	{"main" , evalFnPtr}
	//};

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
	//	EvalAuxData* auxData, Status& s) { return s; }*/
	Status eval(VT& value,
		EvalAuxData* auxData, Status& s) {
		LOG("EVALNODE EVAL on node " + name + " - THIS IS WRONG");
		return s;
	}


	Status preEval(const std::string& stepName, Status& s) {
		// called before each step is run
		// use this to recompile parametres, if they've changed
		return s;
	}

	Status preReset() {
		// before node value is reset in graph
	}
	Status postReset() {
		// after node value is reset in graph
	};
	//EvalGraph<VT>* graphPtr = nullptr;

	EvalGraph<VT>* graph() { return reinterpret_cast<EvalGraph<VT>*>(graphPtr); }

	VT& value() { // retrieve whatever node's current value is in graph
		//return &(graphPtr->results[index]);
		return graph()->results[index];
	}

};

//struct EvalGraph<int>;


struct NodeData {
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
		//LOG("EvalGraph cloneShared");
		return std::shared_ptr<T>(
			static_cast<T*>(T::clone_impl(copyAllResults)));
	}
	//auto cloneShared(bool copyAllResults) const { return std::shared_ptr<T>(dynamic_cast<T*>(clone_impl(copyAllResults))); }
	virtual EvalGraph<VT>* clone_impl(bool copyAllResults) const {
		//LOG("EvalGraph clone impl");
		auto newPtr = new EvalGraph<VT>(*this);
		//newPtr->copyOther(*this, copyAllResults);
		return newPtr;
	};

	virtual void copyOther(const EvalGraph& other, bool copyAllResults = true) {
		//LOG("EVAL GRAPH COPY OTHER, other nodes: " + str(other.nodes.size()))
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
				return nullptr;
		};
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
		LOG("evalNode begin: " + op->name + " " + str(op->index) + " n results: " + str(results.size()));
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
		if (!resultSet) { // if no inputs, copy graph's baseValue
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
	Status evalGraph(Status& s, int upToNodeId = -1, EvalAuxData* auxData = nullptr) {
		/* first input of each node by default should be the manifold stream
		to write on. if a node has no input, it's an ancestor in the graph-
		create an empty manifold object for that object

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
#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "manifold.h"

#include "op.h"

namespace ed {
	/// do we need to keep entire manifolds? can we eval the whole graph live at all times?
	// how does that work with inheriting values and geometry? - if an element op doesn't OVERRIDE the value, that
	// just means the previous one will be used - I think that's the definition of inheritance, right?
	struct StrataOpGraph {
		/* for building graph, use some kind of lock or mutex to modify vectors
		safely from separate maya nodes - after that, graph can be eval'd in parallel safely
		*/


		std::string name; // still not sure how name should be handled

		// assume that ops are added to vectors in order

		std::vector<StrataOp> ops;
		//std::vector<std::unique_ptr<StrataOp>> ops;

		std::unordered_map<std::string, size_t> opNameIndexMap;
		//std::vector<StrataManifold> manifolds;

		std::vector<std::unordered_set<int>> generations;
		bool graphChanged = true; // if true, needs full rebuild of topo generations

		// transient map for topology
		std::unordered_map<int, std::unordered_set<int>> nodeDependentsMap;

		StrataManifold baseManifold; // by default empty manifold

		StrataOp* addOp(StrataOp& op) {
			/*passes ownership of op object to vector,
			returns a new pointer to it*/
			size_t newIndex = ops.size();
			ops.push_back(op);
			StrataOp* newOp = &ops[newIndex];

			//ops.push_back(std::unique_ptr<StrataOp>(&op));
			//StrataOp* newOp = ops[newIndex].get();
			opNameIndexMap[newOp->name] = newIndex;
			graphChanged = true;
			return newOp;
		}

		Status rebuildGraphStructure(Status& s) {
			/* rebuild graph tables and caches from current ops in vector*/
			opNameIndexMap.clear();
			opNameIndexMap.reserve(ops.size());
			for (size_t i = 0; i < ops.size(); i++) {
				opNameIndexMap[ops[i].name] = static_cast<int>(i);
			}
			std::vector<std::unordered_set<int>> result;
			s = getTopologicalGenerations(result, s);
			graphChanged = false;
			return s;
		}
		inline StrataOp* getOp(StrataOp*& opIndex) {
			// included here for similar syntax to get op pointer, no matter the input
			// not sure if this is actually useful in c++
			return opIndex;
		}
		inline StrataOp* getOp(const int opIndex) {
			return &(ops[opIndex]);
		}
		inline StrataOp* getOp(const std::string opName) {
			return &(ops[opNameIndexMap[opName]]);
		}

		inline std::unordered_set<int>* opOutputs(const int opIndex) {
			if (graphChanged) {
				Status s;
				rebuildGraphStructure(s);
			}
			return &nodeDependentsMap[opIndex];
		}

		inline std::unordered_set<int>* opOutputs(const StrataOp* op) {
			return opOutputs(op->index);
		}

		/* functions to get nodes in history and future, 
		TODO: maybe cache these? but they only matter in graph editing, 
		I don't think there's any point in it
		*/
		std::unordered_set<int> opsInHistory(int opIndex) {
			/* flat set of all nodes found in history*/
			std::unordered_set<int> result;
			
			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					StrataOp* checkOp = getOp(checkNodeIndex);
					newToCheck.insert(checkOp->inputs.begin(), checkOp->inputs.end());

					// add the inputs to result this iteration to
					result.insert(checkOp->inputs.begin(), checkOp->inputs.end());
				}
				toCheck = newToCheck;
			}
			return result;
		}

		SmallList<std::unordered_set<int>, 8> opsInHistory(int opIndex, bool returnGenerations) {
			/* generation list of nodes in history*/
			SmallList<std::unordered_set<int>, 8> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					StrataOp* checkOp = getOp(checkNodeIndex);
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

		std::unordered_set<int> opsInFuture(int opIndex) {
			/* flat set of all nodes found in future*/
			std::unordered_set<int> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					std::unordered_set<int>* outputIds = opOutputs(checkNodeIndex);
					newToCheck.insert(outputIds->begin(), outputIds->end());

					// add the outputs to the result of this iteration
					result.insert(outputIds->begin(), outputIds->end());
				}
				toCheck = newToCheck;
			}
			return result;
		}

		SmallList<std::unordered_set<int>, 8> opsInFuture(int opIndex, bool returnGenerations) {
			/* generation list of nodes in future*/
			SmallList<std::unordered_set<int>, 8> result;

			std::unordered_set<int> toCheck = { opIndex };
			while (toCheck.size()) {
				std::unordered_set<int> newToCheck;
				for (int checkNodeIndex : toCheck) {
					std::unordered_set<int>* outputIds = opOutputs(checkNodeIndex);
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

		void nodeInputsChanged(int opIndex) {
			graphChanged = true;
		}

		void nodeSetDirty(int opIndex, bool topoDirty, bool dataDirty) {
			// propagate dirty stuff forwards to all nodes
			// we add bools to the base state - 
			// can't use false arguments to set clean with this function
			StrataOp* seedOp = getOp(opIndex);
			seedOp->topoDirty = seedOp->topoDirty || topoDirty;
			seedOp->dataDirty = seedOp->dataDirty || dataDirty;
			for (int futureIndex : opsInFuture(opIndex)) {
				StrataOp* futureOp = getOp(opIndex);
				futureOp->topoDirty = seedOp->topoDirty || topoDirty;
				futureOp->dataDirty = seedOp->dataDirty || dataDirty;
			}
		}

		int checkLegalInput(int opIndex, int testDriverOpIndex) {
			/* if result is 0 all good
			// if not, error and it's illegal
			check for feedback loops
			*/
			if (opIndex == testDriverOpIndex) { // can't connect to itself
				return 1;
			}
			std::unordered_set<int> futureOps = opsInFuture(opIndex);
			if (futureOps.count(testDriverOpIndex)) {
				// node found in future, illegal feedback loop
				return 2;
			}
			return 0;
		}


		Status getTopologicalGenerations(std::vector<std::unordered_set<int>>& result, Status& s) {
			/* get sets of nodes guaranteed not to depend on each other
			// use for priority in scheduling work, but too restrictive to rely
			// on totally for execution
			( eval-ing generations one-by-one means each will always be waiting on the last node in that generation, which might not be useful)
			*/
			std::unordered_set<int> zeroDegree;
			nodeDependentsMap.clear();

			// first gather all nodes with no inputs
			for (StrataOp& op : ops) {
				op.temp_inDegree = static_cast<int>(op.inputs.size());
				// build topology outputs
				for (int inputId : op.inputs) {
					// add this node to the dependents of all its inputs
					nodeDependentsMap[inputId].insert(op.index);
				}
				if (!op.inputs.size()) {
					zeroDegree.insert(op.index);
				}
			}
			while (zeroDegree.size()) {
				result.push_back(zeroDegree);
				zeroDegree.clear();
				// check over all direct children of current generation
				for (auto& nodeId : result[-1]) {
					// can parallelise this
					StrataOp* node = getOp(nodeId);
					// all outputs of this node - decrement their inDegree by 1
					for (int dependentId : nodeDependentsMap[node->index]) {
						StrataOp* dependent = getOp(dependentId);
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
					StrataOp* node = getOp(nodeId);
					node->temp_generationId = i;
				}
				i += 1;
			}
			//return result;
			return s;
		}


		Status evalOp(StrataOp* op, Status& s) {
			/* encapsulated evaluation for a single op
			* we guarantee that all an op's input nodes will already
			* have been evaluated
			*/

			// reset node state
			op->reset();

			// copy geo into node's result, if it has inputs
			if (op->inputs.size()) {
				StrataOp* mainInput = getOp(op->inputs[0]);
				// copy the manifold result of previous node
				op->result = mainInput->result;
			}

			/*TODO: check for errors in evaluation, somehow - 
			that would mean passing out errors to graph eval function

			...we could hypothetically CHECK an MSTATUS...
			...and hypothetically RETURN_IT...
			*/

			// if params have changed, recompile
			if (op->paramsDirty) {
				s = op->evalParams(op->rootExpNode, s);
				CRSTAT(s);
				op->paramsDirty = false;
			}

			// eval topology if dirty
			if (op->topoDirty) {
				s = (op->result, s);
				CRSTAT(s);
				op->topoDirty = false;
			}
			
			// eval data if dirty
			if (op->dataDirty) {
				s = op->evalData(op->result, s);
				CRSTAT(s);
				op->dataDirty = false;
			}

			// donezo
			return s;
		}

		Status evalOpGraph(Status& s, int upToNodeId = -1) {
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
			if (graphChanged) {
				rebuildGraphStructure(s);
				
			}
			CRMSG(s, "ERROR rebuilding graph structure ahead of graph eval, halting Strata graph");

			bool foundEndNode = false;
			// for now just work in generations
			// go through generations in sequence
			for (auto& generation : generations) {

				// go through each node in generation
				// this is the bit that can be parallelised
				for (auto& nodeId : generation) {
					StrataOp* node = getOp(nodeId);
					// check if anything on node is dirty - if so, continue
					if (!(node->dataDirty + node->topoDirty + node->paramsDirty)) {
						continue;
					}
					// eval whole node
					evalOp(node, s);
					CRMSG(s, "ERROR eval-ing op " + node->name + ", halting Strata graph ");

					// mark outputs dirty, ensure they get eval'd in next generation
					for (int dependentId : nodeDependentsMap[node->index]) {
						StrataOp* dependent = getOp(dependentId);
						dependent->topoDirty = true;
					}
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

	};
}

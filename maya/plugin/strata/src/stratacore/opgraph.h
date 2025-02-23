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

		std::string name; // still not sure how name should be handled

		// assume that ops are added to vectors in order

		std::vector<StrataOp> ops;
		std::unordered_map<std::string, size_t> opNameIndexMap;
		//std::vector<StrataManifold> manifolds;

		std::vector<std::unordered_set<int>> generations;
		bool graphChanged = true;

		// transient map for topology
		std::unordered_map<int, std::set<int>> nodeDependentsMap;

		StrataManifold baseManifold; // by default empty manifold

		StrataOp* addOp(StrataOp& op) {
			size_t newIndex = ops.size();
			ops.push_back(op);
			StrataOp* newOp = &ops[newIndex];
			opNameIndexMap[newOp->name] = newIndex;
			graphChanged = true;
			return newOp;
		}

		inline StrataOp* getOp(const int opIndex) {
			return &(ops[opIndex]);
		}
		inline StrataOp* getOp(const std::string opName) {
			return &(ops[opNameIndexMap[opName]]);
		}

		std::vector<std::unordered_set<int>> getTopologicalGenerations() {
			/* get sets of nodes guaranteed not to depend on each other
			// use for priority in scheduling work, but too restrictive to rely
			// on totally for execution
			( eval-ing generations one-by-one means each will always be waiting on the last node in that generation, which might not be useful)
			*/
			std::vector<std::unordered_set<int>> result;
			std::unordered_set<int> zeroDegree;
			nodeDependentsMap.clear();

			// first gather all nodes with no inputs
			for (StrataOp& op : ops) {
				op.temp_inDegree = static_cast<int>(op.inputs.size());
				// build topology outputs
				for (int inputId : op.inputs) {
					// add this node to the dependents of all its inputs
					nodeDependentsMap[op.index].insert(inputId);
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
			return result;
		}


		void evalOp(StrataOp* op) {
			/* encapsulated evaluation for a single op
			* we guarantee that all an op's input nodes will already
			* have been evaluated
			*/

			// first get the manifold object to work on
			StrataManifold* manifoldPtr;
			if (op->inputs.size()) {
				StrataOp* mainInput = getOp(op->inputs[0]);
				// copy the manifold result of previous node
				manifoldPtr = &(StrataManifold(mainInput->result));
			}
			else {
				manifoldPtr = &(StrataManifold());
			}

			// reset node state
			op->reset();

			// if params have changed, recompile
			if (op->paramsDirty) {
				op->rootExpNode = op->evalParams();
				op->paramsDirty = false;
			}

			// still unhappy at eval'ing both of these all the time
			op->evalTopo(*manifoldPtr);
			op->evalData(*manifoldPtr);

			op->result = *manifoldPtr;

		}

		void evalOpGraph(int upToNodeId = -1) {
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
				generations = getTopologicalGenerations();
				graphChanged = false;
			}

			bool foundEndNode = false;
			// for now just work in generations
			// go through generations in sequence
			for (auto& generation : generations) {

				// go through each node in generation
				for (auto& nodeId : generation) {
					StrataOp* node = getOp(nodeId);
					// check if anything on node is dirty - if so, continue
					if (!(node->dataDirty + node->topoDirty + node->paramsDirty)) {
						continue;
					}
					// eval whole node
					evalOp(node);
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
		}

	};
}

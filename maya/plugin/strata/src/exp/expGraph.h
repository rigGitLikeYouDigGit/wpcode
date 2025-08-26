#pragma once

#include "../dirtyGraph.h"
#include "expAtom.h"
#include "expValue.h"

namespace strata {
	struct StrataManifold;

	namespace expns {


		struct Expression;

		struct ExpAuxData : EvalAuxData {
			StrataManifold* manifold = nullptr;
			ExpStatus* expStatus = nullptr;
			//Expression* exp = nullptr;
			/* TODO: we'll have to parallelise status at some point
			* to parallelise exp execution */

			std::vector<int> expValuesToElements(std::vector<ExpValue>& values, Status& s);
			Status& expValuesToElements(Status& s, std::vector<ExpValue>& values, std::vector<int>& result);
			Status& expValuesToElements(Status& s, ExpValue& value, std::vector<int>& result);

		};



		/* using VECTORS of expValues as value types to support unpacking, slicing more easily -
		feels insane, but also sensible
		otherwise each operator can only produce a single discrete result

		register functions in global scope dict, pull copies of of the
		"compiled" master graph into each node that calls it
		*/

		struct ExpGraph : EvalGraph<std::vector<ExpValue>> {


			using EvalGraph::EvalGraph;

			using VT = std::vector<ExpValue>;

			ExpGraph() {
				//EvalGraph<std::vector<ExpValue>>::
				//	EvalGraph<std::vector<ExpValue>>();
			}

			Expression* exp = nullptr; // owner expression
			template <typename ExpOpT, typename NodeT = ExpOpNode>
			NodeT* addNode(const std::string& name = ""
			)
			{
				VT defaultValue;
				std::string nodeName = name; // I don't know how to set a string to one of 2 options - this is infantile
				if (name == "") { // make a default name out of the op type and the latest index
					int nNodes = static_cast<int>(nodes.size());
					nodeName = ExpOpT::OpName;
					nodeName += "_" + std::to_string(nNodes);
				}
				NodeT* baseResult = EvalGraph::addNode<NodeT>(nodeName, defaultValue, nullptr);
				// add ExpOp unique pointer
				baseResult->expAtomPtr = std::make_unique<ExpOpT>();
				return baseResult;
			}

			void clear() {
				nodes.clear();
				results.clear();
				graphChanged = true;
			}

			ExpOpNode* addResultNode();

			ExpOpNode* getResultNode();


		};

		struct ExpOpNode : EvalNode<std::vector<ExpValue>> {
			// delegate eval function to expAtom, pulling in ExpValue arguments from input nodes
			//ExpGraph* graphPtr;

			std::unique_ptr<ExpAtom> expAtomPtr;
			using EvalFnT = Status(*)(ExpOpNode*, std::vector<ExpValue>&, Status&);
			typedef EvalFnT EvalFnT;
			//static Status evalMain(ExpOpNode* node, std::vector<ExpValue>& value, Status& s);
			//EvalFnT evalFnPtr = evalMain; // pointer to op function - if passed, easier than defining custom classes for everything?

			//static Status eval(ExpOpNode* node, std::vector<ExpValue>& value, EvalAuxData* auxData, Status& s);
			virtual Status eval(
				std::vector<ExpValue>& value,
				EvalAuxData* auxData,
				Status& s
			);

			virtual ExpGraph* getGraphPtr() {
				if (graphPtr == nullptr) {
					return nullptr;
				}
				return static_cast<ExpGraph*>(graphPtr);
			}

			virtual EvalNode* clone_impl() const {
				// deepcopy expAtomPtr
				auto newPtr = static_cast<ExpOpNode*>(EvalNode::clone_impl());
				newPtr->expAtomPtr = expAtomPtr->clone();
				return newPtr;
			};

			using EvalNode::EvalNode;
			//ExpOpNode(const int index, const std::string name) : index(index), name(name) {}
		};

	}
}
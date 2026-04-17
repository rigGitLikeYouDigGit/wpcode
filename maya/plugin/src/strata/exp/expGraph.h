#pragma once

#include "../dirtyGraph.h"
#include "../evalGraph.h"

#include "expValue.h"
//#include "expElCompare.h"
#include "expAtom.h"

// include all exp ops for variant
//#include "assignAtom.h"
//#include "callAtom.h"

//#include "groupAtom.h"
//#include "nameAtom.h"
//#include "plusAtom.h"
//#include "resultAtom.h"

#include "atomVariant.h"

namespace strata {
	struct StrataManifold;

	namespace expns {


		struct Expression;

		/* expressions obviously need knowledge of existing state of manifold,
		ie for closest-point or ray operations

		if we wanted to make it general, have a base layer of just the atoms, then CRTP those on
		derived classes that each override eval()? 

		at that point just duplicate the whole thing, probably less hassle
		*/
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


		//using ExpAtomVariant = std::variant<
		//	AssignAtom,
		//	CallAtom,
		//	ExpAtom,
		//	GroupAtom,
		//	NameAtom,
		//	PlusAtom,
		//	ResultAtom
		//>;

		struct ExpOpNode : EvalNode<
			ExpVT,
			ExpAtomVariant
		> {
			// delegate eval function to expAtom, pulling in ExpValue arguments from input nodes
			//ExpGraph* graphPtr;

			/*std::unique_ptr<ExpAtom> expAtomPtr;
			using EvalFnT = Status(*)(ExpOpNode*, std::vector<ExpValue>&, Status&);
			typedef EvalFnT EvalFnT;*/
			//static Status evalMain(ExpOpNode* node, std::vector<ExpValue>& value, Status& s);
			//EvalFnT evalFnPtr = evalMain; // pointer to op function - if passed, easier than defining custom classes for everything?

			//static Status eval(ExpOpNode* node, std::vector<ExpValue>& value, EvalAuxData* auxData, Status& s);
			//virtual Status eval(
			//	std::vector<ExpValue>& value,
			//	EvalAuxData* auxData,
			//	Status& s
			//);

			using EvalNode::EvalNode;

			using EvalLogicVariant = ExpAtomVariant;
			//EvalLogicVariant logic;

			Status eval(ExpVT& value, ExpAuxData* auxData, Status& s) {
				return std::visit([&](auto& logicImpl) {
					// Cast here, logic types never see void*
					return logicImpl.eval(
						value.first,
						auxData,
						value.second,
						s);
					}, logic);
			}

			//virtual ExpGraph* graph() {
			//	if (graphPtr == nullptr) {
			//		return nullptr;
			//	}
			//	return static_cast<ExpGraph*>(graphPtr);
			//}

			//virtual EvalNode* clone_impl() const {
			//	// deepcopy expAtomPtr
			//	auto newPtr = static_cast<ExpOpNode*>(EvalNode::clone_impl());
			//	newPtr->expAtomPtr = expAtomPtr->clone();
			//	return newPtr;
			//};

			
			//ExpOpNode(const int index, const std::string name) : index(index), name(name) {}
		};


		struct ExpGraph : EvalGraphBase<
			ExpVT,
			ExpOpNode,
			ExpGraph
		> {

			using EvalGraphBase::EvalGraphBase;

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


	}
}
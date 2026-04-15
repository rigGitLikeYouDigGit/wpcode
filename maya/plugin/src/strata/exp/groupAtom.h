#pragma once
#include "expAtom.h"

namespace strata {
	namespace expns {

		struct GroupAtom : PrefixParselet {
			/* parse a set of tokens between brackets -
			* this doesn't add a specific node to the graph, just
			* controls how nodes are added and evaluated
			*/
			static constexpr const char* OpName = "group";

			//GroupAtom() {}
			//virtual GroupAtom* clone_impl() const override { return new GroupAtom(*this); };

			////void copyOther(const AssignAtom& other) {
			////	InfixParselet::copyOther(other);
			////}
			//MAKE_COPY_FNS(GroupAtom)

			Status& parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					//int leftIndex,
					int& outNodeIndex,
					Status& s
				);

			Status& eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s);

			int getPrecedence() {
				return Precedence::CALL;
			}
		};

	}
}
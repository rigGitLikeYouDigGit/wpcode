#pragma once
#include "expAtom.h"

namespace strata {
	namespace expns {

		struct GroupAtom : PrefixParselet {
			/* parse a set of tokens between brackets -
			* this doesn't add a specific node to the graph, just
			* controls how nodes are added and evaluated
			*/
			GroupAtom() {}
			static constexpr const char* OpName = "group";
			virtual GroupAtom* clone_impl() const override { return new GroupAtom(*this); };

			//void copyOther(const AssignAtom& other) {
			//	InfixParselet::copyOther(other);
			//}
			MAKE_COPY_FNS(GroupAtom)

			virtual Status parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					//int leftIndex,
					int& outNodeIndex,
					Status& s
				);

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s);

			virtual int getPrecedence() {
				return Precedence::CALL;
			}
		};

	}
}
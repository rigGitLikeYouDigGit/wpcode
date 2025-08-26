#pragma once

#include "expAtom.h"


namespace strata {
	namespace expns {


		struct AssignAtom : InfixParselet {
			/* atom to assign result to a variable
			* arguments should be { variable name , variable value }
			*/
			AssignAtom() {}
			static constexpr const char* OpName = "assign";

			virtual AssignAtom* clone_impl() const override { return new AssignAtom(*this); };

			void copyOther(const AssignAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(AssignAtom)

				virtual int getPrecedence() {
				return Precedence::ASSGIGNMENT;
			}
			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s);

		};
	}
}
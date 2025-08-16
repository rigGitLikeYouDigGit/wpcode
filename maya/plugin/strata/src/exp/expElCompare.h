#pragma once

#include "expParse.h"

namespace strata{
	namespace expns {

		struct GreaterThanAtom : InfixParselet {
			/* atom to assign result to a variable
			* arguments should be { variable name , variable value }
			*/
			GreaterThanAtom() {}
			static constexpr const char* OpName = "assign";

			virtual GreaterThanAtom* clone_impl() const override { return new GreaterThanAtom(*this); };

			void copyOther(const GreaterThanAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(GreaterThanAtom)

				virtual int getPrecedence() {
				return Precedence::CONDITIONAL;
			}
			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			virtual Status eval(
				std::vector<ExpValue>& argList,
				ExpAuxData* auxData,
				std::vector<ExpValue>& result, Status& s);
		};
	}
}


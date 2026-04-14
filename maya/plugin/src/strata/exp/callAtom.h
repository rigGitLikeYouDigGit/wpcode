#pragma once

#include "expAtom.h"

namespace strata {
	namespace expns {

		constexpr const char* resultCallName = "_OUT";

		struct CallAtom : InfixParselet {
			/* call a function by name, with arguments
			* first in arg list is name of function to call
			*/
			static constexpr const char* OpName = "call";
			CallAtom() {}

			virtual CallAtom* clone_impl() const override { return new CallAtom(*this); };

			void copyOther(const CallAtom& other) {
				InfixParselet::copyOther(other);
			}

			MAKE_COPY_FNS(CallAtom)

				virtual Status parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					int leftIndex,
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

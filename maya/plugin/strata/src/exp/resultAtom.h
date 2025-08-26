#pragma once

#include "expAtom.h"

namespace strata {
	namespace expns {

		struct ResultAtom : InfixParselet {
			/*gathers the final value of an expression, where an explicit assign
			* or return is not found -
			* this node should always be 0 in the graph, and probably won't be
			* included in parsing operations
			*
			* ALSO probably duplication in logic with assign op - could literally simplify
			* everything by assigning to an '_OUT' variable by default
			*/
			static constexpr const char* OpName = "result"; // class name of the operation
			ResultAtom() {}

			virtual ResultAtom* clone_impl() const override { return new ResultAtom(*this); };

			void copyOther(const ResultAtom& other) {
				InfixParselet::copyOther(other);
			}

			MAKE_COPY_FNS(ResultAtom)

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, 
				std::vector<ExpValue>& result, Status& s);
		};

	}
}


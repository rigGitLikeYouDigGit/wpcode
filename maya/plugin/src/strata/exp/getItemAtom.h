#pragma once
#include "expAtom.h"

namespace strata {
    namespace expns {

		struct GetItemAtom : InfixParselet {
			/* access an item by square brackets
			* arg 1 is object to access
			* other args are dimensions for array access
			*/
			static constexpr const char* OpName = "getitem";
			/*GetItemAtom() {}
			virtual GetItemAtom* clone_impl() const override { return new GetItemAtom(*this); };

			void copyOther(const GetItemAtom& other) {
				InfixParselet::copyOther(other);
			}
			//MAKE_COPY_FNS(GetItemAtom)*/
			Status& eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{

				return s;
			}

			int getPrecedence() {
				return Precedence::CALL;
			}
		};


    }
}
#pragma once
#include "expAtom.h"

namespace strata {
    namespace expns {


		struct AccessAtom : InfixParselet {
			/* access an object attribute by name, dot operator - eg
			obj.attribute
			*/

			static constexpr const char* OpName = "access";
			//AccessAtom() {}

			//virtual AccessAtom* clone_impl() const override { return new AccessAtom(*this); };

			//void copyOther(const AccessAtom& other) {
			//	InfixParselet::copyOther(other);
			//}

			//MAKE_COPY_FNS(AccessAtom)
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
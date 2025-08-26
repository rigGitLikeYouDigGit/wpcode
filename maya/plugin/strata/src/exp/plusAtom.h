#pragma once

#include "expAtom.h"


namespace strata {
	namespace expns {


		struct PlusAtom : InfixParselet {
			/* atom to assign result to a variable
			* arguments should be { variable name , variable value }
			*/
			std::string opName = "plus"; // class name of the operation
			PlusAtom() {}
			virtual PlusAtom* clone_impl() const override { return new PlusAtom(*this); };

			void copyOther(const PlusAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(PlusAtom);
				virtual Status eval(std::vector<ExpValue>& argList, 
					ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s);

			virtual int getPrecedence() {
				return Precedence::SUM;
			}
		};
	}
}


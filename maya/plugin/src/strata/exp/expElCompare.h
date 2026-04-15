#pragma once

#include "expParse.h"

namespace strata{
	namespace expns {

		struct GreaterThanAtom : InfixParselet {
			/* atom to create sub element(s) greater than one or more conditions
			*/
			static constexpr const char* OpName = "greaterThan";

			//GreaterThanAtom() {}

			//virtual GreaterThanAtom* clone_impl() const override { return new GreaterThanAtom(*this); };

			//void copyOther(const GreaterThanAtom& other) {
			//	InfixParselet::copyOther(other);
			//}
			//MAKE_COPY_FNS(GreaterThanAtom)

			int getPrecedence() {
				return Precedence::CONDITIONAL;
			}
			Status& parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			Status& eval(
				std::vector<ExpValue>& argList,
				ExpAuxData* auxData,
				std::vector<ExpValue>& result, Status& s);
		};

		struct LessThanAtom : InfixParselet {
			static constexpr const char* OpName = "lessThan";

			/*LessThanAtom() {}

			virtual LessThanAtom* clone_impl() const override { return new LessThanAtom(*this); };

			void copyOther(const LessThanAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(LessThanAtom)*/

			int getPrecedence() {
				return Precedence::CONDITIONAL;
			}
			Status& parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			Status& eval(
				std::vector<ExpValue>& argList,
				ExpAuxData* auxData,
				std::vector<ExpValue>& result, Status& s);
		};
	}
}


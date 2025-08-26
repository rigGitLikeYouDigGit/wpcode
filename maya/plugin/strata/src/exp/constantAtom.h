#pragma once

#include "expAtom.h"
#include "expGraph.h"
#include "expValue.h"

namespace strata {
	namespace expns {


		//struct ConstantAtom : ExpAtom {
		struct ConstantAtom : PrefixParselet {
			/* atom to represent a constant numeric value or string
			*/
			ConstantAtom() {}
			static constexpr const char* OpName = "constant";
			float literalVal = 0.0;
			std::string literalStr;

			virtual ConstantAtom* clone_impl() const override { return new ConstantAtom(*this); };

			void copyOther(const ConstantAtom& other) {
				PrefixParselet::copyOther(other);
				literalVal = other.literalVal;
				literalStr = other.literalStr;
			}

			MAKE_COPY_FNS(ConstantAtom);

			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int& outNodeIndex,
				Status& s
			);

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, 
				std::vector<ExpValue>& result, Status& s);
		};
	}
}
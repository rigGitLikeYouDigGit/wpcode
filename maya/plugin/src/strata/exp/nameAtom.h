#pragma once

#include "expAtom.h"

namespace strata {
	namespace expns {



		struct NameAtom : PrefixParselet { // inspired by python's system - same use for var names or functions, attributes etc?
			static constexpr const char* OpName = "name";
			//std::string strName;

			//NameAtom() {}

			//virtual NameAtom* clone_impl() const override { return new NameAtom(*this); };

			//void copyOther(const NameAtom& other) {
			//	PrefixParselet::copyOther(other);
			//	strName = other.strName;
			//}

			//MAKE_COPY_FNS(NameAtom)

				// depending on use, this name will either be set or retrieved by the next operation in graph
			Status& eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
					std::vector<ExpValue>& result,
					Status& s);
			int getPrecedence() {
				return 0;
			}

			Status& parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int& outNodeIndex,
				Status& s
			);
		};
	}
}
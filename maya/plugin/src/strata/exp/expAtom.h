#pragma once

#include <string>
#include <algorithm>
#include <memory>
#include <typeinfo>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include< fstream >
#include<istream>
#include <vector>
#include <typeinfo>
#include <typeindex>
#include <cassert>

#include <enum.h>
#include "../../containers.h"
#include "../status.h"
#include "../macro.h"
#include "../dirtyGraph.h"
#include "../evalGraph.h"
//#include "../stratacore/manifold.h"

#include "../factory.h"

#include "expLex.h"

namespace strata {
	namespace expns {
		struct Expression; // main holder object 
		struct ExpOpNode;
		struct ExpParser;
		struct ExpGraph;
		struct ExpAuxData;
		struct ExpValue;
		struct ExpStatus;

		//typedef ExpOpNode;

		static int fToInt(float f) {
			/* for use turning float numbers to array indices */
			return static_cast<int>(floor(f + EPS_F));
		}


		struct Precedence {
			static constexpr int ASSIGNMENT = 1;
			static constexpr int CONDITIONAL = 2;
			static constexpr int SUM = 3;
			static constexpr int PRODUCT = 4;
			static constexpr int EXPONENT = 5;
			static constexpr int PREFIX = 6;
			static constexpr int POSTFIX = 7;
			static constexpr int CALL = 8;
		}; // do we add in another level for 'pattern'? filters on element names etc



		/* using VECTORS of expValues as value types to support unpacking, slicing more easily -
		feels insane, but also sensible
		otherwise each operator can only produce a single discrete result

		register functions in global scope dict, pull copies of of the
		"compiled" master graph into each node that calls it
		*/

		// arg values, out values
		using ExpVT = std::pair<std::vector<ExpValue>, std::vector<ExpValue>>;


		struct ExpAtom : EvalLogic {
			/* struct to define an operation as part of an expression
			*/
			int startIndex = -1; // where does this atom start (main index of this atom in the expression)
			std::string srcString = ""; // what text in the expression created this atom (inclusive for function calls)
			static constexpr const char* OpName = "base";

			int getPrecedence() {
				return 0;
			}

			//Status& eval(
			//	void* nodePtr, void* valuePtr, void* auxData, Status& s);

			Status& eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
				std::vector<ExpValue>& result, Status& s);

			/* do:
			auto* value = static_cast<Manifold*>(valuePtr);
			auto* node = static_cast<EvalNode<Manifold, EvalLogicVariant>*>(nodePtr);
			*/
			Status& parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					int& outNodeIndex,
					Status& s
				);

		};


		struct PrefixParselet : ExpAtom {
		
			using ExpAtom::ExpAtom;
		
			Status& parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					int& outNodeIndex,
					Status& s
				);
		};
		struct InfixParselet : ExpAtom
		{

			using ExpAtom::ExpAtom;

			int getPrecedence() {
				return Precedence::SUM;
			}

				Status& parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					int leftIndex,
					int& outNodeIndex,
					Status& s
				);
		};



	}
}
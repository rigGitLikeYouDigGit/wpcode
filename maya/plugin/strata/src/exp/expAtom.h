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

#include "wpshared/enum.h"
#include "../containers.h"
#include "../status.h"
#include "../macro.h"
#include "../dirtyGraph.h"
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



#define MAKE_COPY_FNS(classT)\
		~classT() = default;\
		classT(classT const& other) {\
			copyOther(other);\
		}\
		classT(classT&& other) = default;\
		classT& operator=(classT const& other) {\
			copyOther(other);\
			return *this;\
		}\
		classT& operator=(classT&& other) = default;\


		struct Precedence {
			static constexpr int ASSGIGNMENT = 1;
			static constexpr int CONDITIONAL = 2;
			static constexpr int SUM = 3;
			static constexpr int PRODUCT = 4;
			static constexpr int EXPONENT = 5;
			static constexpr int PREFIX = 6;
			static constexpr int POSTFIX = 7;
			static constexpr int CALL = 8;
		}; // do we add in another level for 'pattern'? filters on element names etc

		struct ExpAtom {
			/* struct to define an operation as part of an expression
			*/
			int startIndex = -1; // where does this atom start (main index of this atom in the expression)
			std::string srcString = ""; // what text in the expression created this atom (inclusive for function calls)
			static constexpr const char* OpName = "base";


			void copyOther(const ExpAtom& other) {
				startIndex = other.startIndex;
				srcString = other.srcString;
			}

			auto clone() const { return std::unique_ptr<ExpAtom>(clone_impl()); }
			template < typename pT>
			auto clone() const { return std::unique_ptr<pT>(static_cast<pT*>(clone_impl())); }
			virtual ExpAtom* clone_impl() const = 0;

			ExpAtom() {}
			virtual ~ExpAtom() = default;
			ExpAtom(ExpAtom const& other) {
				copyOther(other);
			}
			ExpAtom(ExpAtom&& other) = default;
			ExpAtom& operator=(ExpAtom const& other) {
				copyOther(other);
			}
			ExpAtom& operator=(ExpAtom&& other) = default;

			// function to run live in op graph
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
				std::vector<ExpValue>& result,
				Status& s);

			// function to insert this op in graph
			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int& outNodeIndex,
				Status& s
			);

			virtual int getPrecedence() {
				return 0;
			}
		};


		struct PrefixParselet : ExpAtom {
			PrefixParselet() {}
			//virtual ~PrefixParselet() = default;
			void copyOther(const PrefixParselet& other) {
				ExpAtom::copyOther(other);
			}
			virtual PrefixParselet* clone_impl() const override { return new PrefixParselet(*this); };

			MAKE_COPY_FNS(PrefixParselet)


				virtual Status parse(
					ExpGraph& graph,
					ExpParser& parser,
					Token token,
					int& outNodeIndex,
					Status& s
				);
		};
		struct InfixParselet : ExpAtom
		{

			InfixParselet() {}
			virtual int getPrecedence() {
				return Precedence::SUM;
			}

			/* for some reason I couldn't get these to work in maps
			without explicitly defining every single copy function by hand
			great
			love it
			*/
			void copyOther(const InfixParselet& other) {
				ExpAtom::copyOther(other);
			}
			virtual InfixParselet* clone_impl() const override { return new InfixParselet(*this); };

			MAKE_COPY_FNS(InfixParselet)

				virtual Status parse(
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
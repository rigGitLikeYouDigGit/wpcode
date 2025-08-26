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
#include<fstream>
#include<istream>
#include <vector>
#include <typeinfo>
#include <typeindex>
#include <cassert>

#include "../status.h"

namespace strata {
	namespace expns {
		struct ExpValue {
			/* intermediate and result struct produced by operators -
			in this way we allow dynamic typing in expressions

			for array values, is there any value in leaving it as an expression rather than
			evaluated sets? then somehow eval -ing the whole thing?
			that's just what we're doing here anyway, I'm dumb

			need exp state to track which node indices represent what variables?
			which indices were last to modify variable value

			*/
			// if this value is a variable?
			std::string varName;

			// vectors always stored as vec4, matrix always stored as 4x4
			//enum struct Type {
			//	Number, String,
			//	Vector,
			//	Matrix
			//};

			//BETTER_ENUM(Type); 

			/* should we only represent vectors through different shapes in arrays?
			go full numpy with it*/
			//Type t = Type::Number;

			// you can take the python scrub out of python
			// we just use strings for vartypes, makes it easier to declare new types, 
			// check for matching / valid conversions, operations etc




			//SmallList<int, 4> dims;
			std::vector<float> numberVals;
			std::vector<std::string> stringVals;
			// store values in flat arraus

			struct Type {
				static constexpr const char* number = "number";
				static constexpr const char* string = "string";

			};

			std::string t = Type::number;

			void copyOther(const ExpValue& other) {
				t = other.t;
				numberVals = other.numberVals;
				stringVals = other.stringVals;
				//dims = other.dims;
				/*dims.clear();
				dims.swap(other.dims);*/
			}

			ExpValue() {}
			~ExpValue() = default;
			ExpValue(ExpValue&& other) noexcept {
				copyOther(other);
			}
			ExpValue(const ExpValue& other) {
				copyOther(other);
			}
			ExpValue& operator=(const ExpValue& other) {
				copyOther(other);
				return *this;
			}
			ExpValue& operator=(ExpValue&& other) = default;

			Status extend(std::initializer_list<ExpValue> vals);

			std::string printInfo();

		};
	}
}
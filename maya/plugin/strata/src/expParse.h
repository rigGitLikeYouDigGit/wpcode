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
#include <map>
#include <vector>
#include <typeinfo>
#include <typeindex>
#include <cassert>


#include "status.h"

#include "dirtyGraph.h"

#include "factory.h"

/*
dead, dead simple way of parsing a string expression into
a list / tree of operations on tokens

jank shall abound

to start with, functions only support scalars and strings.
scalars stored as floats
toInt() on a value object will curtail a float to nearest integer (for use as index)
will also turn a string numeral value to an integer if possible

always kept as arrays?
maybe have separate arrays in values, in case logic has to change? idk

i don't really know a good way to do polymorphism yet, so for now all operators will
be the same class, and just hold a function pointer to their eval function.

Also makes it easier to source a struct instance than a struct type, based on the found string of that
operator -
in python I'd just have a map of { "+" : PlusOperator } as a type, 
but in C it seems more complicated - we need a type_info object? 
still unsure how all that works

multiple operators with the same token can be declared - this is the same as multiple-dispatch in c
maybe we can also check for types here?


// eg for fn pointers
int multiply(int a, int b) { return a * b; }

int main()
{
	int (*func)(int, int);

	// func is pointing to the multiplyTwoValues function

	func = multiply;

*/
namespace ed {

	struct test {
		static const std::string tag;
		//const static std::string name = "ada";
	};


	//typedef ExpOpNode;

	//template <typename VT>
	struct ExpOpNode;

	struct ExpValue {
		/* intermediate and result struct produced by operators -
		in this way we allow dynamic typing in expressions

		for array values, is there any value in leaving it as an expression rather than
		evaluated sets? then somehow eval-ing the whole thing?
		that's just what we're doing here anyway, I'm dumb
		*/
		std::string strVal;
		float scalarVal;

		// if this value is a variable?
		std::string varName;
	};

	struct ExpStatus {
		/*constant state of overall expression - 
		mainly tracking variables*/
		std::map<std::string, ExpValue> varMap;
	};

	struct ExpAtom {
		/* struct to define an operation as part of an expression
		*/
		int startIndex = -1; // where does this atom start (main index of this atom in the expression)
		std::string srcString = ""; // what text in the expression created this atom (inclusive for function calls)
		std::string opName = "base"; // class name of the operation
		std::string fnName; // if this is a named function, or maths operator

		////ExpAtom() = default;

		virtual Status eval(std::vector<ExpValue>& argList, ExpStatus& expStat, std::vector<ExpValue>& result, Status& s) 
		{
			result = argList;
			return s;
		}
	};

	struct AssignAtom : ExpAtom {
		/* atom to assign result to a variable
		* arguments should be { variable name , variable value }
		*/
		std::string opName = "assign"; // class name of the operation
		std::string fnName; // if this is a named function, or maths operator
		virtual Status eval(std::vector<ExpValue>& argList, ExpStatus& expStat, std::vector<ExpValue>& result, Status& s)
		{
			if (argList.size() == 2) { // check only name of variable and variable value passed
				STAT_ERROR(s, "Can only assign single ExpValue to variable, not 0 or multiple");
			}
			argList[0].varName = argList[1].strVal;
			expStat.varMap[argList[0].varName] = argList[0];
			return s;
		}
	};

	struct ExpGrammar{
		Factory<std::string, ExpAtom> atomFactory;

	};

	static ExpGrammar baseGrammar;
	//static void buildBaseGrammar();



	struct ExpOpNode : EvalNode<ExpValue> {

		ExpAtom expAtom;

	};




}
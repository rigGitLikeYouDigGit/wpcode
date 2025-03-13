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


#include "status.h"

#include "dirtyGraph.h"

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

	//typedef ExpOperator;
	struct ExpOperator;

	struct ExpValue {
		/* intermediate and result struct produced by operators - 
		in this way we allow dynamic typing in expressions
		*/
		std::string strVal;
		float scalarVal;

		// if this value is a variable?
		std::string varName;
	};

	static Status templateOpFn(
		const ExpOperator& op//, 
		//const std::vector<ExpValue>& args, 
		//std::vector<ExpValue>& result
	) {
		return Status();
	}
	typedef Status(*OpFnPtr) (const ExpOperator&);


	struct ExpOperator {
		const std::string name;
		const std::string token;
		int nArgs = 1;
		OpFnPtr opFnPtr = nullptr; // pointer to op function

	};

	struct ExpGrammar {
		std::unordered_map< std::string, ExpOperator> opMap;

		ExpOperator* addOp(
			/* tried doing this in the constructor, but the ownership
			got really messy
			*/
			ExpOperator newOp
		) {
			//opMap[newOp.token] = newOp;
			opMap.insert(std::make_pair(newOp.token, newOp));
			return &(opMap[newOp.token]);
		}

	};

	// build default grammar for strata expressions
	static ExpGrammar mainGrammar;
	static ExpOperator* plusOp = mainGrammar.addOp(
		ExpOperator{"plusOp", "+", 2, &templateOpFn}
	);





}
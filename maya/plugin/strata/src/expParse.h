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

	/*static Status templateOpFn(
		ExpOpNode* op, 
		ExpValue& value, 
		Status& s
	) {
		return s;
	}*/


	struct ExpOpNode : EvalNode<ExpValue> {
	//template<typename VT>
	//struct ExpOpNode : EvalNode<VT> {
		//using EvalNode<ExpValue>::EvalNode<ExpValue>; // brother what is this 
		//using EvalFnT = Status(*)(ExpOpNode<VT>*, VT&, Status&);
		///using EvalNode::EvalNode; // brother what is this 
		/*const std::string token;
		int nArgs = 1;*/

		/*
		I GOT A REAL FUNCTION POINTER FOR YOU
		
		I POINT

		YOU FUNCTION
		*/

		typedef Status(*EvalFnT)(ExpOpNode*, ExpValue&, Status&);
		static Status evalMain(ExpOpNode* node, ExpValue& value, Status& s) { return s; }
		EvalFnT evalFnPtr = evalMain; // pointer to op function - if passed, easier than defining custom classes for everything?

		// ordered map to define evaluation order of steps
		std::map<const std::string, EvalFnT&> evalFnMap{
			{"main" , evalFnPtr}
		};



	};

	//// register individual functions against tokens?
	//// seems easier than registering different types, that's super hard in c++
	////typedef  ExpOpNode<ExpValue>::EvalFnT ExpOpFnT; // doesn't match
	////typedef  Status(*)(EvalNode<ExpValue>*, ExpValue&, Status&) ExpOpFnT; // not allowed
	////using ExpOpFnT = Status(*)(EvalNode<ExpValue>*, ExpValue&, Status&);
	////typedef ExpOpFnT ExpOpFnT; // screw this language man

	//typedef Status(*ExpOpFnT)(ExpOpNode*, ExpValue&, Status&);

	struct ExpGrammar {
		std::unordered_map< const std::string, 
			//Status(*)(EvalNode<ExpValue>*, ExpValue&, Status&)
			ExpOpNode::EvalFnT
		> tokenFnMap;

		void registerOpFn(const std::string token, 
			//ExpOpFnT &opFn
			ExpOpNode::EvalFnT opFn
			//Status(*)(EvalNode<ExpValue>*, ExpValue&, Status&) opFn
		) {
			tokenFnMap[token] = opFn;
			//tokenFnMap.insert(token, &opFn);
			//tokenFnMap.emplace(token, opFn);  // error C2064: term does not evaluate to a function taking 1 arguments
			////// to hell with all of this man
		}
	};



	//extern ExpGrammar mainGrammar;
	//
	////Status assignOpFn(EvalNode<ExpValue>* node, ExpValue& value, Status& s) {
	//Status assignOpFn(ExpOpNode* node, ExpValue& value, Status& s) {
	//	return s;
	//}

	//static ExpOpFnT fnPtr = &assignOpFn;

	//static void initMainGrammar() {
	//	
	//	mainGrammar.registerOpFn("=", &assignOpFn);

	//}


	//};

	// build default grammar for strata expressions
	




}
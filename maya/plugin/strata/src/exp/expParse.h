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

#include "expAtom.h"
#include "expGraph.h"

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

separate parsing from eval -
in exp graph to pull and assign variables, have nodes to assign and pull from status, by string var name

then to eval exp with changing global values, update values in the status and eval the graph as is

$("value") -> get or set variable 
$(side + "Nose") -> variable by dynamic name

python semantics by default, everything is a reference

how do we do "eval"

eval() is a function node
holds its own internal graph - exports a list of variables from the outer scope that it depends on?

if a dynamic-lookup variable is passed in to the eval code string, then it's untrackable, and we assume we have
to recompile the sub-expression every run



eArm(pShoulder, pElbow, pWrist, pHandEnd) // make a new edge
eArm(pShoulder, pElbow, pWrist, pHandEnd,
	s=eArmSpaceCrv) // make a new edge with a given parent space




// eg for fn pointers
int multiply(int a, int b) { return a * b; }

int main()
{
	int (*func)(int, int);

	// func is pointing to the multiplyTwoValues function

	func = multiply;

*/
namespace strata {
	struct StrataManifold;

	namespace expns {


		struct test {
			static const std::string tag;
			//const static std::string name = "ada";
		};

		static void test() {
			//span<const char> strView;
		}




		struct ExpStatus {
			/*constant state of overall expression -
			mainly tracking variables*/
			std::map<std::string, ExpValue> varMap = {};
			// map of which node to pull from for any variable name - 
			// updated by eval whenever var is modified
			std::map<std::string, int> varIndexMap = {};

			// we specialise this for strata - this will never be its own separate library anyway
			//StrataManifold* manifold = nullptr;

			/* status has to be copied out by every node too : (otherwise races ?
			// not necessarily, only during parsing, otherwise graph shape 
			ensures vars are always pulled/modified in right order?

			what about UNKNOWN / DYNAMIC variable names?
			not yet
			*/
		};


		/* below is an adaption of Bob Nystrom's Bantam parser, as best as I understand
		it
		apologies in advance

		Bantam returns the new Expression from each function, to be included in subsequent
		parselets - here we have an output int reference for the index of the new node added to the graph

		*/


		struct ExpParseStatus {
			/* small struct used to track global information on parsing - 
			no support yet for dynamic stuff
			*/
			// map of which node to pull from for any variable name - 
			// updated by parsing whenever var is modified
			std::map<std::string, int> varIndexMap;

		};



		struct AccessAtom : InfixParselet {
			/* access an object attribute by name, dot operator - eg
			obj.attribute
			*/
			
			static constexpr const char* OpName = "access";
			AccessAtom() {}

			virtual AccessAtom* clone_impl() const override { return new AccessAtom(*this); };

			void copyOther(const AccessAtom& other) {
				InfixParselet::copyOther(other);
			}

			MAKE_COPY_FNS(AccessAtom)
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{

				return s;
			}
			virtual int getPrecedence() {
				return Precedence::CALL;
			}
		};

		struct GetItemAtom : InfixParselet {
			/* access an item by square brackets
			* arg 1 is object to access
			* other args are dimensions for array access
			*/
			static constexpr const char* OpName = "getitem";
			GetItemAtom() {}
			virtual GetItemAtom* clone_impl() const override { return new GetItemAtom(*this); };

			void copyOther(const GetItemAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(GetItemAtom)
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{

				return s;
			}

			virtual int getPrecedence() {
				return Precedence::CALL;
			}
		};





		struct ExpParser {
			/* holds intermediate list of tokens and index for current position
			used in place of the full Parser object from Nystrom's version.
			Also includes the functionality of his tokenIterator, since we use
			separate lexing passes beforehand
			*/
			int index = 0;
			std::vector<Token> parsedTokens = {};
			std::vector<Token> readTokens = {};
			std::vector<Token> mRead = {};
			std::unordered_map<Token::Kind, std::unique_ptr<InfixParselet>> mInfixParselets = {};
			std::unordered_map<Token::Kind, std::unique_ptr<PrefixParselet>> mPrefixParselets = {};


			//virtual ~ExpParser() = default;


			void copyOther(const ExpParser& other) {
				index = other.index;
				parsedTokens = other.parsedTokens;
				readTokens = other.readTokens;
				mRead = other.mRead;
				// deep copy of parselet maps
				// eventually see about making some kind of "clonable" base or interface for this sort of thing
				mPrefixParselets.clear();
				mPrefixParselets.reserve(other.mPrefixParselets.size());
				for (auto& p : other.mPrefixParselets) {
					//std::unique_ptr<InfixParselet> ptr = p.second.get()->clone();  //NB - cannot implicitly cast up to container of derived, from pointer of base
					mPrefixParselets.insert(
						std::make_pair(p.first, 
							(p.second->clone<PrefixParselet>())
						)
					);
				}
				for (auto& p : other.mInfixParselets) {
					//std::unique_ptr<InfixParselet> ptr = p.second.get()->clone();  //NB - cannot implicitly cast up to container of derived, from pointer of base
					mInfixParselets.insert(
						std::make_pair(p.first,
							(p.second->clone<InfixParselet>())
						)
					);
				}
			}

			ExpParser();

			MAKE_COPY_FNS(ExpParser)

			void registerParselet(Token::Kind token, std::unique_ptr<PrefixParselet> parselet) {
				mPrefixParselets.insert(std::make_pair(token, std::move(parselet)));
			}

			void registerParselet(Token::Kind token, std::unique_ptr<InfixParselet> parselet) {
				mInfixParselets.insert(std::make_pair(token, std::move(parselet)));
			}

			void resetTokens(std::vector<Token>& aParsedTokens) {
				/* restart parser to work on a new set of tokens*/
				//LOG("resetTokens");
				parsedTokens = aParsedTokens;
				readTokens.clear();
				mRead.clear();
			}

			virtual bool hasNext() const { return true; }
			virtual Token next() {
				if (index < parsedTokens.size()) {
					index += 1;
					return parsedTokens[index - 1];
				}
				return Token(Token::Kind::End);
			}

			Token consume(Token::Kind expected, Status& s) {
				//LOG("consume, expected: " + Token::kindStrStatic(expected));
				Token token = lookAhead(0);
				if (token.getKind() != expected) {
					/*throw ParseException("Expected token " + tokentype::toString(expected) +
						" and found " + tokentype::toString(token.getType()));*/
					CWMSG(s, "unexpected token type found in consume()");
					s.val += 1;
				}
				return consume();
			}

			Token consume() {
				// Make sure we've read the token.
				//LOG("consume empty");
				lookAhead(0);

				Token front = mRead.front();
				mRead.erase(mRead.begin());
				return front;
			}

			bool match(Token::Kind expected) {
				Token token = lookAhead(0);
				if (token.getKind() != expected) {
					return false;
				}

				consume();
				return true;
			}

			Status parseExpression(
				ExpGraph& graph,
				int& outNodeIndex,
				int precedence
			);
			Status parseExpression(
				ExpGraph& graph,
				int& outNodeIndex

			) {
				return parseExpression(
					graph,
					outNodeIndex,
					0
				);
			}
			Token lookAhead(unsigned int distance) {
				// Read in as many as needed.
				while (distance >= mRead.size()) {
					mRead.push_back(next());
				}

				// Get the queued token.
				return mRead[distance];
			}

			int getPrecedence() {
				auto k = lookAhead(0);
				auto kKind = k.getKind();
				/*auto itParser = mInfixParselets.find(
					lookAhead(0).getKind());*/
				auto itParser = mInfixParselets.find(kKind);
				if (itParser != mInfixParselets.end()) {
					return itParser->second->getPrecedence();
				}

				return 0;
			}
		};



		struct Expression {
			/* master container for individual expression*/
			std::string srcStr;

			ExpParseStatus parseStatus;
			Lexer lexer;
			ExpGraph graph;
			ExpParser parser;
			bool needsRecompile = true;
			///// ok so what if EVERYTHING was in the same graph
			// interesting but not right now, different value types get super annoying
			//std::unique_ptr<ExpStatus> globalExpStatusPtr; // owned status for variables, built up as graph is parsed. pointer to allow passing in custom types for different graph types
			// status doesn't need to be stored by expression

			Expression() : lexer("") {
			}

			Expression(std::string inSrcStr) : lexer(inSrcStr.c_str()), graph() {
				srcStr = inSrcStr;
			}
			Expression(const char* inSrcStr) : lexer(inSrcStr), graph() {
				srcStr = inSrcStr;
			}

			void copyOther(const Expression& other);

			~Expression() = default; Expression(Expression const& other) {
				copyOther(other);
			} Expression(Expression&& other) = default; 
			Expression& operator=(Expression const& other) {
				copyOther(other);
				return *this;
			} 
			Expression& operator=(Expression&& other) = default;;


			void setSource(const char* inSrcStr);

			Status parse();

			Status result(std::vector<ExpValue>*& outResult,
				//ExpStatus* expStatus,
				ExpAuxData* auxData
			);


		};
	}
}
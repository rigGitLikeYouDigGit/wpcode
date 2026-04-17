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

#include <variant>
#include <optional>

#include <enum.h>
#include "../../containers.h"
#include "../status.h"
#include "../macro.h"
#include "../dirtyGraph.h"
//#include "../stratacore/manifold.h"

#include "../factory.h"
#include "expLex.h"

#include "expAtom.h"
#include "expGraph.h"

#include "atomVariant.h"


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
	s=eArmSpaceCrv) // make a new edge with a given domain space




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

		struct ParseletRegistry {
			std::unordered_map<Token::Kind, PrefixParseletVariant> prefixParselets;
			std::unordered_map<Token::Kind, InfixParseletVariant> infixParselets;

			// Register a prefix parselet
			template<typename T>
			void registerPrefix(Token::Kind token, T&& parselet) {
				static_assert(std::is_constructible_v<PrefixParseletVariant, T>,
					"Type must be a valid prefix parselet");
				prefixParselets[token] = std::forward<T>(parselet);
			}

			// Register an infix parselet
			template<typename T>
			void registerInfix(Token::Kind token, T&& parselet) {
				static_assert(std::is_constructible_v<InfixParseletVariant, T>,
					"Type must be a valid infix parselet");
				infixParselets[token] = std::forward<T>(parselet);
			}

			// Lookup methods
			std::optional<PrefixParseletVariant> findPrefix(Token::Kind token) const {
				auto it = prefixParselets.find(token);
				if (it != prefixParselets.end() && !std::holds_alternative<std::monostate>(it->second)) {
					return it->second;  
				}
				return std::nullopt;
			}

			std::optional<InfixParseletVariant> findInfix(Token::Kind token) const {
				auto it = infixParselets.find(token);
				if (it != infixParselets.end() && !std::holds_alternative<std::monostate>(it->second)) {
					return it->second;
				}
				return std::nullopt;
			}
		};


// precedence from any parselet
		struct GetPrecedenceVisitor {
			int operator()(std::monostate) const { return 0; }

			template<typename T>
			int operator()(const T& parselet) const {
				return parselet.getPrecedence();
			}
		};

		// Parse using prefix parselet
		struct PrefixParseVisitor {
			ExpGraph& graph;
			ExpParser& parser;
			Token token;
			int& outNodeIndex;
			Status& s;

			Status operator()(std::monostate) {
				STAT_ERROR(s, "No prefix parselet registered for token");
				return s;
			}

			template<typename T>
			Status operator()(T& parselet) {
				return parselet.parse(graph, parser, token, outNodeIndex, s);
			}
		};

		// Parse using infix parselet
		struct InfixParseVisitor {
			ExpGraph& graph;
			ExpParser& parser;
			Token token;
			int leftIndex;
			int& outNodeIndex;
			Status& s;

			Status operator()(std::monostate) {
				STAT_ERROR(s, "No infix parselet registered for token");
				return s;
			}

			template<typename T>
			Status operator()(T& parselet) {
				return parselet.parse(graph, parser, token, leftIndex, outNodeIndex, s);
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
			ParseletRegistry registry;

			//virtual ~ExpParser() = default;


			void copyOther(const ExpParser& other);

			//ExpParser();

			//MAKE_COPY_FNS(ExpParser)
			void initializeParselets();

			template<typename T>
			void registerPrefixParselet(Token::Kind token, T parselet = T{}) {
				registry.registerPrefix(token, std::move(parselet));
			}

			template<typename T>
			void registerInfixParselet(Token::Kind token, T parselet = T{}) {
				registry.registerInfix(token, std::move(parselet));
			}

			//void registerParselet(Token::Kind token, std::unique_ptr<PrefixParselet> parselet);

			//void registerParselet(Token::Kind token, std::unique_ptr<InfixParselet> parselet);

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
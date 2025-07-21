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
#include "../stratacore/manifold.h"

#include "../factory.h"
#include "expLex.h"

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




// eg for fn pointers
int multiply(int a, int b) { return a * b; }

int main()
{
	int (*func)(int, int);

	// func is pointing to the multiplyTwoValues function

	func = multiply;

*/
namespace ed {

	namespace expns {

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


		struct test {
			static const std::string tag;
			//const static std::string name = "ada";
		};

		static void test() {
			//span<const char> strView;
		}


		struct Expression; // main holder object 
		struct ExpOpNode;
		struct ExpParser;
		struct ExpGraph;

		//typedef ExpOpNode;

		static int fToInt(float f) {
			/* for use turning float numbers to array indices */
			return static_cast<int>(floor(f + EPS_F));
		}



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

			struct Type {
				static constexpr const char* number = "number";
				static constexpr const char* string = "string";

			};

			std::string t = Type::number;




			//SmallList<int, 4> dims;
			std::vector<float> numberVals;
			std::vector<std::string> stringVals;
			// store values in flat arraus

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

			Status extend(std::initializer_list<ExpValue> vals) {
				/* flatten all given values into this one - types must match*/
				Status s;
				for (auto& el : vals) {
					if (!(el.t == t)) {
						STAT_ERROR(s, "Flattening base value of type: TODO passed different type: TODO ");
					}
				}
				for (auto& el : vals) {
					numberVals.insert(numberVals.end(), el.numberVals.begin(), el.numberVals.end());
					stringVals.insert(stringVals.end(), el.stringVals.begin(), el.stringVals.end());
				}
				return s;
			}

			std::string printInfo() {
				std::string result = "<expValue- varName:" + varName + " >";
				return result;
			}

		};

		/* SHOULD THIS OBJECT LINK TO MANIFOLD?
		OR
		SHOULD HIGHER EXPAUXDATA

		flat better than nested until we find a required case
		*/
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

		struct ExpParseStatus {
			/* small struct used to track global information on parsing - 
			no support yet for dynamic stuff
			*/
			// map of which node to pull from for any variable name - 
			// updated by parsing whenever var is modified
			std::map<std::string, int> varIndexMap;

		};

		struct Expression;

		struct ExpAuxData : EvalAuxData {
			StrataManifold* manifold = nullptr;
			ExpStatus* expStatus = nullptr;
			//Expression* exp = nullptr;
			/* TODO: we'll have to parallelise status at some point
			* to parallelise exp execution */

			std::vector<int> expValuesToElements(std::vector<ExpValue>& values, Status& s);
		};


		/* below is an adaption of Bob Nystrom's Bantam parser, as best as I understand
		it
		apologies in advance

		Bantam returns the new Expression from each function, to be included in subsequent
		parselets - here we have an output int reference for the index of the new node added to the graph

		*/


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

			ExpAtom(){}
			virtual ~ExpAtom() = default;
			ExpAtom(ExpAtom const& other) {
				copyOther(other);
			}
			ExpAtom(ExpAtom&& other) = default;
			ExpAtom& operator=( ExpAtom const& other) {
				copyOther(other);
			}
			ExpAtom& operator=(ExpAtom&& other) = default;

			// function to run live in op graph
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
				std::vector<ExpValue>& result,
				Status& s)
			{
				LOG("EXPATOM base eval - probably wrong");
				result = argList;
				return s;
			}

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


		/* using VECTORS of expValues as value types to support unpacking, slicing more easily -
		feels insane, but also sensible
		otherwise each operator can only produce a single discrete result

		register functions in global scope dict, pull copies of of the
		"compiled" master graph into each node that calls it
		*/

		struct ExpGraph : EvalGraph<std::vector<ExpValue>> {
			

			using EvalGraph::EvalGraph;

			using VT = std::vector<ExpValue>;

			ExpGraph() {
				//EvalGraph<std::vector<ExpValue>>::
				//	EvalGraph<std::vector<ExpValue>>();
			}

			Expression* exp = nullptr; // owner expression
			template <typename ExpOpT, typename NodeT = ExpOpNode>
			NodeT* addNode(const std::string& name = ""
			)
			{
				VT defaultValue;
				std::string nodeName = name; // I don't know how to set a string to one of 2 options - this is infantile
				if (name == "") { // make a default name out of the op type and the latest index
					int nNodes = static_cast<int>(nodes.size());
					nodeName = ExpOpT::OpName;
					nodeName += "_" + std::to_string(nNodes);
				}
				NodeT* baseResult = EvalGraph::addNode<NodeT>(nodeName, defaultValue, nullptr);
				// add ExpOp unique pointer
				baseResult->expAtomPtr = std::make_unique<ExpOpT>();
				return baseResult;
			}

			void clear() {
				nodes.clear();
				results.clear();
				graphChanged = true;
			}

			ExpOpNode* addResultNode();

			ExpOpNode* getResultNode();


		};

		struct ExpOpNode : EvalNode<std::vector<ExpValue>> {
			// delegate eval function to expAtom, pulling in ExpValue arguments from input nodes
			//ExpGraph* graphPtr;

			std::unique_ptr<ExpAtom> expAtomPtr;
			using EvalFnT = Status(*)(ExpOpNode*, std::vector<ExpValue>&, Status&);
			typedef EvalFnT EvalFnT;
			//static Status evalMain(ExpOpNode* node, std::vector<ExpValue>& value, Status& s);
			//EvalFnT evalFnPtr = evalMain; // pointer to op function - if passed, easier than defining custom classes for everything?

			//static Status eval(ExpOpNode* node, std::vector<ExpValue>& value, EvalAuxData* auxData, Status& s);
			virtual Status eval(
				std::vector<ExpValue>& value,
				EvalAuxData* auxData,
				Status& s
			);

			virtual ExpGraph* getGraphPtr() {
				if (graphPtr == nullptr) { 
					return nullptr; }
				return static_cast<ExpGraph*>(graphPtr);
			}

			virtual EvalNode* clone_impl() const { 
				// deepcopy expAtomPtr
				auto newPtr = static_cast<ExpOpNode*>(EvalNode::clone_impl());
				newPtr->expAtomPtr = expAtomPtr->clone();
				return newPtr;
			};

			using EvalNode::EvalNode;
			//ExpOpNode(const int index, const std::string name) : index(index), name(name) {}
		};

		struct PrefixParselet : ExpAtom{
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

			MAKE_COPY_FNS(ConstantAtom)

			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int& outNodeIndex,
				Status& s
			) {
				
				ExpOpNode* newNode = graph.addNode<ConstantAtom>();
				outNodeIndex = newNode->index;
				ConstantAtom* op = static_cast<ConstantAtom*>(newNode->expAtomPtr.get());
				switch (token.getKind()) {
					case Token::Kind::String: {
						op->literalStr = token.lexeme();	}
					case Token::Kind::Number: {	
						op->literalVal = std::stof(token.lexeme());	}
					default: {
						STAT_ERROR(s, "Unknown token kind to Constant Atom, halting");	}
				}
				return s;
			}

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{
				ExpValue v;
				if (!literalStr.empty()) {
					v.stringVals = { literalStr };
				}
				else { v.numberVals = { literalVal }; }
				//v.dims = { 1 };
				result.push_back(v);
				return s;
			}
		};

		struct NameAtom : PrefixParselet { // inspired by python's system - same use for var names or functions, attributes etc?
			NameAtom() {}
			static constexpr const char* OpName = "name";
			std::string strName;

			virtual NameAtom* clone_impl() const override { return new NameAtom(*this); };

			void copyOther(const NameAtom& other) {
				PrefixParselet::copyOther(other);
				strName = other.strName;
			}

			MAKE_COPY_FNS(NameAtom)

			// depending on use, this name will either be set or retrieved by the next operation in graph
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData,
				std::vector<ExpValue>& result, 
				Status& s)
			{
				ExpValue v;
				v.stringVals = { strName };
				v.varName = strName;
				result.push_back(v);
				return s;
			}
			virtual int getPrecedence() {
				return 0;
			}

			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int& outNodeIndex,
				Status& s
			);
		};

		struct AssignAtom : InfixParselet {
			/* atom to assign result to a variable
			* arguments should be { variable name , variable value }
			*/
			AssignAtom() {}
			static constexpr const char* OpName = "assign";

			virtual AssignAtom* clone_impl() const override { return new AssignAtom(*this); };

			void copyOther(const AssignAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(AssignAtom)

			virtual int getPrecedence() {
				return Precedence::ASSGIGNMENT;
			}
			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{
				/*TODO: could allow multiple values here for concatenation - 
				in that case we would only need to check that types match between them
				*/
				if (!(argList.size() == 2)) { // check only name of variable and variable value passed
					STAT_ERROR(s, "Can only assign single ExpValue to variable, not 0 or multiple");
				}
				//argList[0].varName = argList[1].strVal;
				ExpValue v;
				v.varName = argList[0].stringVals[0];
				v.copyOther(argList[1]);

				// create variable in expression status / scope
				auxData->expStatus->varMap[v.varName] = v;

				// copy left-hand into this node's result, as the value of this variable at this moment
				result.insert(result.begin(), argList.begin() + 1, argList.end());
				return s;
			}

		};

		struct GroupAtom : PrefixParselet {
			/* parse a set of tokens between brackets - 
			* this doesn't add a specific node to the graph, just
			* controls how nodes are added and evaluated
			*/
			GroupAtom() {}
			static constexpr const char* OpName = "group";
			virtual GroupAtom* clone_impl() const override { return new GroupAtom(*this); };

			//void copyOther(const AssignAtom& other) {
			//	InfixParselet::copyOther(other);
			//}
			MAKE_COPY_FNS(GroupAtom)

			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				//int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{
				return s;
			}

			virtual int getPrecedence() {
				return Precedence::CALL;
			}
		};

		constexpr const char* resultCallName = "_OUT";

		struct CallAtom : InfixParselet {
			/* call a function by name, with arguments
			* first in arg list is name of function to call
			*/
			static constexpr const char* OpName = "call";
			CallAtom() {}

			virtual CallAtom* clone_impl() const override { return new CallAtom(*this); };

			void copyOther(const CallAtom& other) {
				InfixParselet::copyOther(other);
			}

			MAKE_COPY_FNS(CallAtom)

			virtual Status parse(
				ExpGraph& graph,
				ExpParser& parser,
				Token token,
				int leftIndex,
				int& outNodeIndex,
				Status& s
			);

			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{
				if (!(argList.size() == 2)) { // check only name of variable and variable value passed
					STAT_ERROR(s, "Can only assign single ExpValue to variable, not 0 or multiple");
				}
				//argList[0].varName = argList[1].strVal;
				ExpValue v;
				v.varName = argList[0].stringVals[0];
				v.copyOther(argList[1]);

				// create variable in expression status / scope
				auxData->expStatus->varMap[v.varName] = v;

				return s;
			}

			virtual int getPrecedence() {
				return Precedence::CALL;
			}
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


		struct ResultAtom : InfixParselet {
			/*gathers the final value of an expression, where an explicit assign
			* or return is not found -
			* this node should always be 0 in the graph, and probably won't be
			* included in parsing operations
			* 
			* ALSO probably duplication in logic with assign op - could literally simplify
			* everything by assigning to an '_OUT' variable by default
			*/
			static constexpr const char* OpName = "result"; // class name of the operation
			ResultAtom() {}

			virtual ResultAtom* clone_impl() const override { return new ResultAtom(*this); };

			void copyOther(const ResultAtom& other) {
				InfixParselet::copyOther(other);
			}

			MAKE_COPY_FNS(ResultAtom)
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{
				if (argList.size() == 0) { // nothing to do
					return s;
				}
				// check that all incoming arguments have the same type
				std::string firstType = argList[0].t;
				//for (auto& arg : argList) {
				for (int i = 1; i < argList.size(); i++) {
					ExpValue arg = argList[i];
					if (arg.t != firstType) {
						STAT_ERROR(s, "Mismatch in value types for result, halting");
					}
					argList[0].extend({ arg });
				}

				//result.push_back(argList[0]);
				return s;
			}
		};

		struct PlusAtom : InfixParselet {
			/* atom to assign result to a variable
			* arguments should be { variable name , variable value }
			*/
			std::string opName = "plus"; // class name of the operation
			PlusAtom() {}
			virtual PlusAtom* clone_impl() const override { return new PlusAtom(*this); };

			void copyOther(const PlusAtom& other) {
				InfixParselet::copyOther(other);
			}
			MAKE_COPY_FNS(PlusAtom)
			virtual Status eval(std::vector<ExpValue>& argList, ExpAuxData* auxData, std::vector<ExpValue>& result, Status& s)
			{
				if (!(argList.size() == 2)) { // check only name of variable and variable value passed
					STAT_ERROR(s, "Can only add 2 values together ");
				}

				ExpValue v;

				v.copyOther(argList[1]);

				// create variable in expression status / scope
				auxData->expStatus->varMap[v.varName] = v;

				return s;
			}

			virtual int getPrecedence() {
				return Precedence::SUM;
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
				auto itParser = mInfixParselets.find(
					lookAhead(0).getKind());
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
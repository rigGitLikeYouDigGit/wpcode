
#include "expParse.h"
using namespace ed;
using namespace ed::exp;

const std::string test::tag("hello");

//ExpGrammar baseGrammar;



Status ExpOpNode::evalMain(ExpOpNode* node, std::vector<ExpValue>& value, Status& s) {
	/* pull in ExpValues from input nodes, join all arguments together - 
	MAYBE there's a case for naming blocks of arguments but that gets insane - 
	python kwargs seem a bit excessive for now

	pass into atom as arguments*/
	std::vector<ExpValue> arguments;


	ExpGraph* testGraphPtr = reinterpret_cast<ExpGraph*>((node->graphPtr));
	if (testGraphPtr == nullptr) {
		STAT_ERROR(s, "UNABLE TO CAST NODE GRAPHPTR TYPE TO EXPGRAPH*");
	}

	ExpGraph* graphPtr = node->getGraphPtr();
	for (int index : node->inputs) {
		arguments.insert(arguments.end(), 
			graphPtr->results[index].begin(),
			graphPtr->results[index].end());
	}

	Status result = node->expAtomPtr->eval(
		arguments,
		testGraphPtr->exp,
		value, 
		s);
	return s;
}

//Status parseLine(std::string srcLine, Expression* e, int& resultNodeId) {
//	// parse a single line or enclosed span, set the result id to the node in the graph to pass on
//	// if brackets/ sub scope encountered, recurse 
//
//	//std::stack<Token> stack;
//
//}

Status validateAndParseStrings(std::string srcStr, std::vector<Token>& parsedTokens) {
	Status s;
	//lexer = Lexer(srcStr.c_str());
	Lexer lexer(srcStr.c_str());

	//std::stack<Token> quoteTokenStack;
	std::vector<Token> quoteTokenStack;
	//std::stack<Token> parsedTokenStack;

	for (Token& token = lexer.next();
		//!(token.is_one_of(Token::Kind::End, Token::Kind::Unexpected));
		true;
		token = lexer.next())
	{
		if (token.kind() == Token::Kind::Unexpected) {
			STAT_ERROR(s, "ERROR in parsing expression:\n source string: " + srcStr +
				"\nError at char " + std::to_string(lexer.index()) + ", unexpected token: " + token.lexeme());
			return s;
		}

		if (token.kind() == Token::Kind::End) {

			// if a quote is on the stack, there is an unterminated string
			if (quoteTokenStack.size()) {
				STAT_ERROR(s,
					"ERROR in parsing expression:\n source string: " + srcStr +
					"\nError at char " + std::to_string(lexer.index()) + ", unclosed quotes "
				);
			}

			return s;
		}

		// check if this token is a quote - if so:
		// - if it matches the last on the quote stack, pop off the last, as it forms a full string
		// - otherwise, the add quote token to quote stack, and add it to the string being formed

		if (token.is_one_of(Token::Kind::SingleQuote, Token::Kind::DoubleQuote)) {

			// if quote stack is empty, begin new string
			if (quoteTokenStack.size() == 0) {
				quoteTokenStack.push_back(token);
				//parsedTokens.push(Token(Token::Kind::String)); // new string token, does not contain the quote that began it
				parsedTokens.push_back(Token(Token::Kind::String)); // new string token, does not contain the quote that began it
				continue;
			}
			if (token.kind() == quoteTokenStack[quoteTokenStack.size() - 1].kind()) { // matches, pop off the quote
				//quoteTokenStack.pop();
				quoteTokenStack.pop_back();
			}
			else {
				// add this new quote to the stack
				//quoteTokenStack.push(token);
				quoteTokenStack.push_back(token);
			}
			///* if there are still quotes in stack, we just popped off an internal quote -
			//ie
			//" outer ' inner ' end "
			//we just capped off inner, so still need to append it to the string token
			//*/
			//if (quoteTokenStack.size()) {
			//	parsedTokenStack.top().append(token.lexeme());
			//	continue;
			//}
		}

		// check if a quote is currently open
		if (quoteTokenStack.size()) {
			// if yes, we add the current token's lexeme to the top of the parsed stack
			//parsedTokens.top().append(token.lexeme());
			parsedTokens.back().append(token.lexeme());
			continue;
		}
		parsedTokens.push_back(token);
	}
	return s;
}



Status parseScope(std::vector<Token> scopeTokens, Expression& exp,
	std::vector<int>& outNodeIndices) {
	/* runs over contained contents of brackets - 
	populate a vector of ints for the resolved, top-level elements of this scope*/
	Status s;
	std::vector<int> tokenStack; // build up stack of nodes for operations
	std::vector<int> scopeStack; // indices of nodes that 
	for (int i = 0; i < scopeTokens.size(); i++)
	{
		Token token = scopeTokens[i];
		ExpOpNode* newNode = nullptr;
		switch (token.kind()) {
		case Token::Kind::String: // add a literal quoted string
		{
			newNode = exp.graph.addNode<ConstantAtom>();
			ConstantAtom* op = static_cast<ConstantAtom*>(newNode->expAtomPtr.get());
			op->literalStr = token.lexeme();
		}
		case Token::Kind::Number: // add a literal number (stored as float by default)
		{
			newNode = exp.graph.addNode<ConstantAtom>();
			ConstantAtom* op = static_cast<ConstantAtom*>(newNode->expAtomPtr.get());
			op->literalVal = std::stof(token.lexeme());
		}
		case Token::Kind::Dollar:
			// set up a variable name or pattern
			continue;

		case Token::Kind::Identifier: {
			// check if this identifier is a variable, preceded by a dollar
			if (i && (scopeTokens[i - 1].kind() == Token::Kind::Dollar)) {
				continue;
			}
			// create a Name op, add it to the stack
			newNode = exp.graph.addNode<NameAtom>();
			NameAtom* op = static_cast<NameAtom*>(newNode->expAtomPtr.get());
			op->strName = token.lexeme();
			tokenStack.push_back(newNode->index);
		}
									/// CALLING FUNCTIONS - on hitting a right-paren, check back in stack to corresponding left
		case Token::Kind::LeftParen: {
			//if(parsedTokens[i-1] == )
		}
		default:
			continue;
		}
		if (newNode == nullptr) {
			STAT_ERROR(s, "UNKNOWN TOKEN: " + token.lexeme() + " - halting");
		}
		newNode->expAtomPtr.get()->srcString = token.lexeme();




	}
	return s;
}

Status Expression::parse() {
	/* update internal structures, return a status on completion or error
	* 
	* FIRST step is to check for quoted strings, since those might contain any
	* characters including newlines, colons, etc - wrap them in single String token
	* 
	* then split to lines
	* 
	* first check through for strings, that might contain semicolons - after that, split to separate lines
	* something like a braced/scope object? same thing for strngs? appending
	* 
	* 
	* use stack of expression status objects as scopes?
	* 
	* if final top level of expression has no assign and no return, we assume the whole thing
	* to be the expression result - pass it to the result op of the graph
	*  
	* I am glad Djikstra is long dead.
	* this concentrated cringe would surely kill him
	*/
	
	std::vector<Token> parsedTokens;
	Status s = validateAndParseStrings(srcStr, parsedTokens);
	if (s) { return s; }

	graph.clear();
	graph.addResultNode();

	
}


//

//

//
////ExpGrammar mainGrammar;
////
////mainGrammar.registerOpFn()
////
////
////Status assignOpFn(ExpOperator<ExpValue>* node, ExpValue& value, Status& s) {
////	return s;
////}
//
//
//
//
//	/* demo below from Mohit on stack overflow
//	still trying to understand it
//	*/
//
//void fun1(void) {
//	std::cout << "inside fun1\n";
//}
//
//int fun2() {
//	std::cout << "inside fun2\n";
//	return 2;
//}
//
//int fun3(int a) {
//	std::cout << "inside fun3\n";
//	return a;
//}
//
//std::vector<int> fun4() {
//	std::cout << "inside fun4\n";
//	std::vector<int> v(4, 100);
//	return v;
//}
//
//// every function pointer will be stored as this type
//typedef void (*voidFunctionType)(void);
//
//struct Interface {
//
//	std::map<std::string, std::pair<voidFunctionType, std::type_index>> m1;
//
//	template<typename T>
//	void insert(std::string s1, T f1) {
//		auto tt = std::type_index(typeid(f1));
//		m1.insert(std::make_pair(s1,
//			std::make_pair((voidFunctionType)f1, tt)));
//	}
//
//	template<typename T, typename... Args>
//	T searchAndCall(std::string s1, Args&&... args) {
//		auto mapIter = m1.find(s1);
//		/*chk if not end*/
//		auto mapVal = mapIter->second;
//
//		// auto typeCastedFun = reinterpret_cast<T(*)(Args ...)>(mapVal.first); 
//		auto typeCastedFun = (T(*)(Args ...))(mapVal.first);
//
//		//compare the types is equal or not
//		assert(mapVal.second == std::type_index(typeid(typeCastedFun)));
//		return typeCastedFun(std::forward<Args>(args)...);
//	}
//};
//
//int main() {
//	Interface a1;
//	a1.insert("fun1", fun1);
//	a1.insert("fun2", fun2);
//	a1.insert("fun3", fun3);
//	a1.insert("fun4", fun4);
//
//	a1.searchAndCall<void>("fun1");
//	int retVal = a1.searchAndCall<int>("fun3", 2);
//	a1.searchAndCall<int>("fun2");
//	auto temp = a1.searchAndCall<std::vector<int>>("fun4");
//
//	return 0;
//}

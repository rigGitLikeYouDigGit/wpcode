
#include "expParse.h"

#include "../logger.h"
using namespace ed;
using namespace ed::expns;

const std::string test::tag("hello");

//ExpGrammar baseGrammar;



//Status ExpOpNode::evalMain(ExpOpNode* node, std::vector<ExpValue>& value, Status& s) {
Status ExpOpNode::eval(ExpOpNode* node, std::vector<ExpValue>& value, EvalAuxData* auxData, Status& s) {
	/* pull in ExpValues from input nodes, join all arguments together - 
	MAYBE there's a case for naming blocks of arguments but that gets insane - 
	python kwargs seem a bit excessive for now

	pass into atom as arguments*/
	std::vector<ExpValue> arguments;

	
	if (node->graphPtr == nullptr) {
		STAT_ERROR(s, "UNABLE TO CAST NODE GRAPHPTR TYPE TO EXPGRAPH*");
	}
	//ExpGraph* testGraphPtr = static_cast<ExpGraph*>((node->graphPtr));
	//ExpGraph* testGraphPtr = reinterpret_cast<ExpGraph*>((node->graphPtr));
	/*if (testGraphPtr == nullptr) {
		STAT_ERROR(s, "UNABLE TO CAST NODE GRAPHPTR TYPE TO EXPGRAPH*");
	}*/

	ExpGraph* graphPtr = node->getGraphPtr();
	for (int index : node->inputs) {
		arguments.insert(arguments.end(), 
			graphPtr->results[index].begin(),
			graphPtr->results[index].end());
	}

	Status result = node->expAtomPtr->eval(
		arguments,
		graphPtr->exp,
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
		if (token.getKind() == Token::Kind::Unexpected) {
			STAT_ERROR(s, "ERROR in parsing expression:\n source string: " + srcStr +
				"\nError at char " + std::to_string(lexer.index()) + ", unexpected token: " + token.lexeme());
			return s;
		}

		if (token.getKind() == Token::Kind::End) {

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
			if (token.getKind() == quoteTokenStack[quoteTokenStack.size() - 1].getKind()) { // matches, pop off the quote
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


// function to insert this op in graph
Status ExpAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int& outNodeIndex,
	Status& s
) {
	return s;
}

// function to insert this op in graph
Status PrefixParselet::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int& outNodeIndex,
	Status& s
) {
	return s;
}

Status InfixParselet::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	return s;
}

Status CallAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	// check if name of function is the graph's result node - 
	// add each separate argument as expression results
	DirtyNode* newNode = nullptr;
	DirtyNode* leftNode = graph.getNode(leftIndex);
	if (leftNode->name == resultCallName) {
		DEBUGS("found call result node in graph, using for top-level return");
		newNode = graph.getResultNode();
	}
	else {
		newNode = graph.addNode<CallAtom>();
		outNodeIndex = newNode->index;

		// add name of function to call inputs
		newNode->inputs.push_back(leftIndex);
	}
	// parse arg lists and add to input list
	if (!parser.match(Token::Kind::RightParen)) {
		do {
			int argIndex = -1;
			s = parser.parseExpression(graph, argIndex);
			CWRSTAT(s, "error parsing arg from CallAtom, halting");
			newNode->inputs.push_back(argIndex);
		} while (parser.match(Token::Kind::Comma));
		parser.consume(Token::Kind::RightParen, s);
		CWRSTAT(s, "error finding rightParen for callAtom");
	}
	return s;
}


Status AssignAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	int right;
	s = parser.parseExpression(graph, right, Precedence::ASSGIGNMENT - 1);
	CWRSTAT(s, "error parsing right side of assignment: " + token.lexeme());
	DirtyNode* newNode = graph.addNode<AssignAtom>();

	// left index should be a name, and right index should be the value
	ExpOpNode* leftNode = static_cast<ExpOpNode*>(graph.getNode(leftIndex));
	NameAtom* leftOp = dynamic_cast<NameAtom*>(leftNode->expAtomPtr.get());
	if (!leftOp) {
		STAT_ERROR(s, "error reinterpreting left op in assignment, could not recover NameAtom from input node");
	}
	newNode->inputs.push_back(leftIndex);
	newNode->inputs.push_back(right);

	// set the index of this variable in the parse state, as this node is most recent
	// to modify it

	graph.exp->parseStatus.varIndexMap[leftOp->strName] = newNode->index;
	outNodeIndex = newNode->index;

	return s;
}



Status GroupAtom::parse(
	ExpGraph& graph,
	ExpParser& parser,
	Token token,
	//int leftIndex,
	int& outNodeIndex,
	Status& s
) {
	s = parser.parseExpression(graph, outNodeIndex);
	parser.consume(Token::Kind::RightParen, s);
	return s;
} 

Status ExpParser::parseExpression(
	ExpGraph& graph,
	int& outNodeIndex,
	int precedence
) {
	Status s;
	Token token = consume();
	auto it = mPrefixParselets.find(token.getKind());

	if (it == mPrefixParselets.end())
		//throw ParseException("Could not parse \"" + token.getText() + "\".");
		STAT_ERROR(s, "Could not find prefixParselet for token: " + token.lexeme() + " , halting");

	PrefixParselet* prefix = it->second.get();
	Status parseS;
	prefix->parse(graph,  *this, token, outNodeIndex, parseS);
	CWRSTAT(parseS, "error parsing prefix ");

	while (precedence < getPrecedence()) {
		token = consume();

		int outInfixNodeIndex = -1;
		InfixParselet* infix = mInfixParselets[token.getKind()].get();
		Status infixS;
		infixS = infix->parse(
			graph,
			*this,
			token,
			int(outInfixNodeIndex),
			outInfixNodeIndex, 
			infixS);
		CWRSTAT(infixS, "error parsing infix")

		// set the input on the given node
		graph.getNode(outInfixNodeIndex)->inputs.push_back(outNodeIndex);
		graph.nodeInputsChanged(outInfixNodeIndex);
		// no idea if this is right
		outNodeIndex = outInfixNodeIndex;
	}
	return s;
}

ExpOpNode* ExpGraph::addResultNode() {
	return addNode<ResultAtom, ExpOpNode>(resultCallName);
}

ExpOpNode* ExpGraph::getResultNode() {
	return static_cast<ExpOpNode*>(getNode(0));
}

std::vector<int> ExpAuxData::expValuesToElements(std::vector<ExpValue>& values, Status& s) {
	/* resolve all possible values to elements */
	LOG("expValuesToElements: " + str(values.size()));
	std::vector<int> result;
	for (size_t vi = 0; vi < values.size(); vi++) {
		//for (auto& v : values) {
		ExpValue& v = values[vi];
		l("check value:" + v.printInfo());
		for (auto& f : v.numberVals) { // check for integer indices
			int id = fToInt(f);
			SElement* ptr = manifold->getEl(id);
			if (ptr == nullptr) { // index not found in manifold
				continue;
			}
			if (!seqContains(result, id)) { // add unique value found
				result.push_back(id);
			}
		}
		for (auto& s : v.stringVals) { // check for string names
			// patterns will already have been expanded by top level
			SElement* ptr = manifold->getEl(s);
			if (ptr == nullptr) {
				continue;
			}
			if (!seqContains(result, ptr->globalIndex)) { // add unique value found
				result.push_back(ptr->globalIndex);
			}
		}
	}
	return result;
}

Status Expression::parse() {
	/* update internal structures, return a status on completion or error
	* I am glad Djikstra is long dead.
	* this concentrated cringe would surely kill him
	*
	* if wrapInResultCall, the expression is wrapped as "RESULT(" exp ")"
	* this way we don't have to deal with multiple values at the top level?
	*
	*/
	Status s;
	if (srcStr == "") {
		STAT_ERROR(s, "cannot parse empty source string for expression, halting");
	}

	std::vector<Token> parsedTokens;
	 s = validateAndParseStrings(srcStr, parsedTokens);
	if (s) { return s; }

	parseStatus = ExpParseStatus();

	graph.clear();
	graph.addResultNode();

	parser.resetTokens(parsedTokens);

	int outIndex = -1;
	Status parseS = parser.parseExpression(graph, outIndex);
	return parseS;
}

Status Expression::result( 
	std::vector<ExpValue>*outResult,
	//ExpStatus * expStatus,
	ExpAuxData* auxData

) {
	// get result of expression at top level
	Status s;
	LOG("exp RESULT: " + srcStr);
	if (srcStr == "") { // no expression string, empty result
		//STAT_ERROR(s, "cannot parse empty source string for expression, halting");
		return s;
	}
	if(needsRecompile){
		s = parse();
		CWRSTAT(s, "could not recompile expression " + srcStr + " to get value, halting");
		return s;
	}

	graph.evalGraph(s, graph.getResultNode()->index, auxData);
	CWRSTAT(s, "error evaluating expression: " + srcStr + " , halting");
	outResult = &graph.results[graph.getResultNode()->index];
	return s;
}


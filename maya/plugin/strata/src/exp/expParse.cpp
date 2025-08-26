
#include "expParse.h"
#include "expValue.h"

#include "../stratacore/manifold.h" // fine to tightly couple here, this expression language is never meant to be standalone

#include "../logger.h"

#include "expElCompare.h"

#include "assignAtom.h"
#include "callAtom.h"
#include "constantAtom.h"
#include "groupAtom.h"
#include "nameAtom.h"
#include "resultAtom.h"



using namespace strata;
using namespace strata::expns;


const std::string test::tag("hello");


/* largely copied from https://github.com/jwurzer/bantam-cpp
modified to produce an evalGraph after parsing
*/




Status validateAndParseStrings(std::string srcStr, std::vector<Token>& parsedTokens) {
	/* check for syntax errors I guess? I won't lie, I have no memory of this entire file
	*/
	Status s;
	LOG("validate and parse: " + srcStr + ", " + str(parsedTokens));
	//lexer = Lexer(srcStr.c_str());
	Lexer lexer(srcStr.c_str());

	//std::stack<Token> quoteTokenStack;
	std::vector<Token> quoteTokenStack;
	//std::stack<Token> parsedTokenStack;
	int limit = 100;
	int i = 0;
	for (Token& token = lexer.next();
		//!(token.is_one_of(Token::Kind::End, Token::Kind::Unexpected));
		true;
		token = lexer.next())
	{
		if (token.getKind() == Token::Kind::End) {
			l("hit end token");

			// if a quote is on the stack, there is an unterminated string
			if (quoteTokenStack.size()) {
				STAT_ERROR(s,
					"ERROR in parsing expression:\n source string: " + srcStr +
					"\nError at char " + std::to_string(lexer.index()) + ", unclosed quotes "
				);
			}

			return s;
		}

		/* skip whitespaces */
		if (token.is_one_of(Token::Kind::Space, Token::Kind::Space)) {
			continue;
		}

		i += 1;
		if (i > limit) {
			l("PARSE LIMIT REACHED, BREAKING");
			break;
		}
		l("parse token:" + str(token));
		if (token.getKind() == Token::Kind::Unexpected) {
			STAT_ERROR(s, "ERROR in parsing expression:\n source string: " + srcStr +
				"\nError at char " + std::to_string(lexer.index()) + ", unexpected token: " + token.lexeme());
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






ExpParser::ExpParser() {
	registerParselet(Token::Kind::Identifier, std::make_unique<NameAtom>());
	registerParselet(Token::Kind::String, std::make_unique<ConstantAtom>());
	registerParselet(Token::Kind::Number, std::make_unique<ConstantAtom>());
	registerParselet(Token::Kind::LeftParen, std::make_unique<GroupAtom>());
	registerParselet(Token::Kind::LessThan, std::make_unique<LessThanAtom>());
	registerParselet(Token::Kind::GreaterThan, std::make_unique<GreaterThanAtom>());

}

Status ExpParser::parseExpression(
	ExpGraph& graph,
	int& outNodeIndex,
	int precedence
) {
	Status s;
	LOG("parser parseExpression");
	
	Token token = consume();
	while (token.getKind() == Token::Kind::Space) {
		token = next();
		token = consume();
	}
	auto it = mPrefixParselets.find(token.getKind());
	if (it == mPrefixParselets.end()) {
		l("prefix not found for token:" + token.kindStr() + ", returning");
		STAT_ERROR(s, "Could not find prefixParselet for token: " + token.lexeme() + " , halting"); 
	}

	DirtyNode* topNode = graph.getNode(outNodeIndex); // get top node beforehand

	PrefixParselet* prefix = it->second.get(); // strName is empty
	Status parseS;
	prefix->parse(graph,  *this, token, outNodeIndex, parseS); // remove copy here if possible
	CWRSTAT(parseS, "error parsing prefix ");
	
	//graph.setOutputNode(0);
	//graph.getNode(graph.getOutputIndex())->inputs.push_back(outNodeIndex);
	//graph.getNode(outNodeIndex)->inputs.push_back(outNodeIndex);



	int limit = 100;
	int i = 0;
	while (precedence < getPrecedence()) {
		i += 1;
		if (i > limit) {
			l("HIT LIMIT, BREAKING");
			STAT_ERROR(s, "HIT PRECEDENCE LIMIT");
		}
		token = consume();
		l("parse token:" + str(token));
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

	/* add final node to inputs of top?*/
	if (topNode != nullptr) {
		topNode->inputs.push_back(outNodeIndex);
	}


	return s;
}

ExpOpNode* ExpGraph::addResultNode() {
	return addNode<ResultAtom, ExpOpNode>(resultCallName);
}

ExpOpNode* ExpGraph::getResultNode() {
	return static_cast<ExpOpNode*>(getNode(0));
}


Status& ExpAuxData::expValuesToElements(Status& s, ExpValue& value, std::vector<int>& result) {
	for (auto& f : value.numberVals) { // check for integer indices
		int id = fToInt(f);
		SElement* ptr = manifold->getEl(id);
		if (ptr == nullptr) { // index not found in manifold
			continue;
		}
		if (!seqContains(result, id)) { // add unique value found
			result.push_back(id);
		}
	}
	for (auto& s : value.stringVals) { // check for string names
		// patterns will already have been expanded by top level
		SElement* ptr = manifold->getEl(s);
		if (ptr == nullptr) {
			continue;
		}
		if (!seqContains(result, ptr->globalIndex)) { // add unique value found
			result.push_back(ptr->globalIndex);
		}
	}
	return s;

}

Status& ExpAuxData::expValuesToElements(Status& s, std::vector<ExpValue>& values, std::vector<int>& result) {
	if (!values.size()) {
		return s; }
	for (size_t vi = 0; vi < values.size(); vi++) {
		ExpValue& v = values[vi];
		s = expValuesToElements(s, v, result);
	}
	return s;
}

std::vector<int> ExpAuxData::expValuesToElements(std::vector<ExpValue>& values, Status& s) {
	/* resolve all possible values to elements */
	
	std::vector<int> result;
	if (!values.size()) { return result; }
	///LOG("expValuesToElements: " + str(values.size()));
	for (size_t vi = 0; vi < values.size(); vi++) {
		//for (auto& v : values) {
		ExpValue& v = values[vi];
		//l("check value:" + v.printInfo());
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

void Expression::setSource(const char* inSrcStr) {
	// only recompile if string has changed
	//LOG("setSource: " + std::string(inSrcStr) + ", was " + std::string(srcStr));
	if (strcmp(inSrcStr, srcStr.c_str()) == 0) { return; }
	//l("string is new, setting");
	srcStr = inSrcStr;
	lexer = Lexer(inSrcStr);
	needsRecompile = true;
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
	//LOG("PARSE:" + srcStr)
	Status s;
	if (srcStr == "") {
		STAT_ERROR(s, "cannot parse empty source string for expression, halting");
	}

	std::vector<Token> parsedTokens;
	 //s = validateAndParseStrings(srcStr, parsedTokens);
	// try something new to capture multiple top-level values
	 s = validateAndParseStrings( "(" + srcStr + ")", parsedTokens);
	 CWRSTAT(s, "ERROR in validateParseStrings");

	parseStatus = ExpParseStatus();
	graph = ExpGraph();
	graph.exp = this;
	//graph.clear();
	//graph.addResultNode();

	parser.resetTokens(parsedTokens);

	int outIndex = -1;
	Status parseS = parser.parseExpression(graph, outIndex);
	return parseS;
}

Status Expression::result( 
	std::vector<ExpValue>*& outResult,
	//ExpStatus * expStatus,
	ExpAuxData* auxData

) {
	// get result of expression at top level
	Status s;
	//LOG("exp RESULT: " + srcStr);
	if (srcStr == "") { // no expression string, empty result
		//STAT_ERROR(s, "cannot parse empty source string for expression, halting");
		return s;
	}
	if(needsRecompile){
		//l("needs recompile");
		s = parse();
		CWRSTAT(s, "could not recompile expression " + srcStr + " to get value, halting");
		//return s;
	}
	//l("eval-ing graph");
	//graph.evalGraph(s, graph.getResultNode()->index, auxData);
	graph.evalGraphSerial(s, graph.getResultNode()->index, auxData);
	CWRSTAT(s, "error evaluating expression: " + srcStr + " , halting");
	outResult = &graph.results[graph.getResultNode()->index];
	return s;
}


void Expression::copyOther(const Expression& other) {
	//LOG("EXP copyOther: " + srcStr + " from " + other.srcStr);
	srcStr = other.srcStr;
	parseStatus = other.parseStatus;
	lexer = other.lexer;
	graph = other.graph;
	needsRecompile = other.needsRecompile;
}
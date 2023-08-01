
from __future__ import annotations
import typing as T

import ast
import re
from dataclasses import dataclass
from ast import unparse

from wplib.ast import parseStrToASTModule, astModuleToExpression
from wplib.object import TypeNamespace
from wplib.expression.constant import MASTER_GLOBALS_EVALUATOR_KEY
from wplib.expression.error import CompilationError, ExpSyntaxError

"""functions checking expression syntax for user input
"""

def getBracketContents(s:str, encloseChars="()")->tuple[tuple, str]:
	"""recursively get items contained in outer brackets
	couldn't find a flexible way online so doing it raw
	"""
	stack = 0
	openIndex = 0
	closeIndex = 0
	result = []
	for i, char in enumerate(s):
		if char == encloseChars[0]:
			if stack == 0:
				openIndex = i
			stack += 1
		elif char == encloseChars[1]:
			stack -= 1
			if stack == 0:
				result.append(s[closeIndex:openIndex])
				result.append(getBracketContents(s[ openIndex + 1 : i ], encloseChars))
				closeIndex = i + 1
	result.append(s[closeIndex:])
	#return tuple(filter(None, result))
	return tuple(i for i in result if not (i == ""))

# print(getBracketContents(testStr))
# print(getBracketContents(invalidStr))

def bracketContentsAreFunction(expBracketContents:tuple[tuple, str]) ->bool:
	"""check if expression is a function
	- does expression start with brackets?
	- does first bracket group precede a colon?
	"""
	if not isinstance(expBracketContents[0], tuple): # flat string
		return False
	if expBracketContents[1][0] != ":": # no colon after first bracket group
		return False
	return True

def restoreBracketContents(expBracketContents:tuple[tuple, str], encloseChars="()")->str:
	"""restore bracket contents to original string"""
	result = ""
	for i, item in enumerate(expBracketContents):
		if isinstance(item, tuple):
			result += encloseChars[0] + restoreBracketContents(item, encloseChars) + encloseChars[1]
		else:
			result += item
	return result


def textDefinesExpFunction(text: str) -> bool:
	"""check if text defines a function
	- does expression start with brackets?
	- does first bracket group precede a colon?
	"""
	return bracketContentsAreFunction(getBracketContents(text.strip()))


def splitTextBodySignature(text: str) -> tuple[str, str]:
	"""split function text into body and signature"""
	bracketParts = getBracketContents(text.strip())
	signature = bracketParts[0]
	body = bracketParts[1:]
	return restoreBracketContents(signature), restoreBracketContents(body)[1:]


# framework for declaring sequence of passes run over expression string -
# this seems like a good balance between efficiency and flexibility.
# I've been struggling with this system for so long, getting something
# on the board is good enough for now

class CustomSyntaxPass(ast.NodeTransformer):
	""" ABSTRACT

	run any custom rules over a raw expression string -
	favour defining and running multiple of these objects
	as separate passes over expression, for separate rules
	of syntax.

	Raise ExpSyntaxError to indicate a syntax error in this pass -
	give as much information as possible, since the order of passes
	may be important.

	A pass may plausibly operate only on a raw string, on an AST,
	or even the reparsed version of it.

	"""

	def __init__(self):
		super(CustomSyntaxPass, self).__init__()
		self.currentExpGlobals = {}

	def _getSyntaxLocalMap(self)->dict:
		"""return a map of any custom syntax to be replaced
		before parsing to ast"""
		return {}

	def preProcessRawString(self, s:str)->str:
		"""run any operations on the raw expression string before parsing to ast
		like converting illegal characters and tokens
		override here"""
		return s


	# def processExpression(self, expInput:str, expGlobals:dict, returnAST=True)->(ast.Expr, str):
	# 	"""run any syntax operations on string,
	# 	then parse it to an ast tree using custom syntax converter
	# 	inputGlobals will be modified
	# 	"""
	# 	print("process exp", expInput, self)
	# 	self.currentExpGlobals = dict(expGlobals)
	# 	# modify raw string if needed, removing illegal characters
	# 	processedString = self.preProcessRawString(expInput)
	#
	# 	# parse to AST
	# 	inputASTModule = parseStrToASTModule(processedString)
	# 	inputASTExt = astModuleToExpression(inputASTModule)
	#
	# 	# process AST
	# 	processedAST = self.visit(inputASTExt)
	# 	ast.fix_missing_locations(processedAST)





def _not_fn(element):
	"""substitute for 'not' where supported
	"""
	try:
		return element.__not__()
	except AttributeError:
		return not element


def _and_fn(element, other):
	"""substitute for 'not' where supported
	"""
	try:
		return element.__and__(other)
	except AttributeError:
		return element and other


def _or_fn(element, other):
	"""substitute for 'not' where supported
	"""
	try:
		return element.__or__(other)
	except AttributeError:
		return element or other


def _in_fn(element, container):
	"""substitute this on any instance of 'in' keyword"""
	try:
		return container.__contains__(element)
	except AttributeError: # fall back if __contains__ has not been defined
		return element in container

class SyntaxKeywordFunctionPass(CustomSyntaxPass):
	"""forces 'and', 'or' etc keywords to explicitly call their
	corresponding magic methods on objects involved -
	otherwise some object derived behaviour can kick in first

	eg :

	"a" and "b" -> __and__fn("a", "b")

	"""

	fnMap = {ast.In : _in_fn,
			 ast.And : _and_fn,
			 ast.Or : _or_fn,
			 ast.Not : _not_fn
			 }
	def _getSyntaxLocalMap(self)->dict:
		"""return map of {function name : function} to
		update expression locals """
		fnMap = {i.__name__ : i for i in self.fnMap.values()}
		return fnMap

	def visit_UnaryOp(self, node: T.UnaryOp) -> T.Any:
		if type(node.op) in self.fnMap:
			newFnTree = ast.Call(
				func=ast.Name(id=self.fnMap[type(node.op)].__name__, ctx=ast.Load()),
				args=[node.operand],
				keywords=[])
			return newFnTree
		return node
	def visit_Compare(self, node: ast.Compare):
		if type(node.ops[0]) in self.fnMap:
			newFnTree = ast.Call(
				func=ast.Name(id=self.fnMap[type(node.ops[0])].__name__, ctx=ast.Load()),
				args=[
					self.visit(node.left),
					self.visit(node.comparators[0])],
					keywords=[])
			return newFnTree
		return node
	def visit_BoolOp(self, node: T.BoolOp) -> T.Any:
		if type(node.op) in self.fnMap:
			newFnTree = ast.Call(
				func=ast.Name(id=self.fnMap[type(node.op)].__name__, ctx=ast.Load()),
				args=[
					self.visit(node.values[0]),
					self.visit(node.values[1])],
					keywords=[])
			return newFnTree
		return node
	#endregion
	pass



nullExpChar = "####"
reMasterSplitStr = r"[ \s+ \+ \- \* \/ \% \( \)]"
reMasterPattern = re.compile(reMasterSplitStr)
reSearchMasterPattern = re.compile(r"[A-Za-z0-9_]+")

class ExpTokens(TypeNamespace):

	class ExpressionToken(TypeNamespace.base()):
		"""class for an expression item,
		identified by a leading (and or trailing) character
		eg:

		$p , @p , @(a + b) etc

		it's possible this could be done with a single dataclass,
		inheritance is my crutch
		"""

		head = nullExpChar # by default a token that will never appear
		tail = nullExpChar

		def __init__(self, content:str):
			self.content = content


	class At(ExpressionToken):
		head = "@"


	class Dollar(ExpressionToken):
		head = "$"


	class PointyBracket(ExpressionToken):
		head = "<"
		tail = ">"


	class Tilde(ExpressionToken):
		head = "~"


	class Hash(ExpressionToken):
		head = "#"


	class Percent(ExpressionToken):
		head = "%"


	class Ampersand(ExpressionToken):
		head = "&"


	class Backquote(ExpressionToken):
		head = "`"
		tail = "`"


class SyntaxTokenReplacerPass(CustomSyntaxPass):
	"""transformer to replace custom tokens with functions to resolve them
	"""

	splitChar = "_eyye_"

	def __init__(self,
	             tokenTypes=(ExpTokens.At,
	                         ExpTokens.Dollar,
	                         ExpTokens.PointyBracket)
				 ):
		super(SyntaxTokenReplacerPass, self).__init__()
		self.tokenTypes = tokenTypes
		self.tokenBaseCharMap = {}
		self.tokenLegalCharMap = {}

	def _getSyntaxLocalMap(self)->dict:
		"""return dict of {name : value} to update expression globals"""
		return {"_" + i.__name__ : i for i in self.tokenTypes}

	def preProcessRawString(self, s:str) ->str:
		"""look up any token characters in string - replace them with
		function calls to resolve those strings with evaluator.

		eg:
		@p + $p -> __evaluator__.resolveToken(_AtToken_("p")) + __evaluator__.resolveToken(_DollarToken_("p"))

		"""

		# build local token map
		for tokenType in self.tokenTypes:
			self.tokenBaseCharMap[(tokenType.head, tokenType.tail)] = tokenType
			legalChars = (f"{tokenType.__name__}_HEAD" + self.splitChar,
						  self.splitChar + f"{tokenType.__name__}_TAIL")
			self.tokenLegalCharMap[tokenType] = legalChars

		# replace chars
		for tokenType, replaceChars in self.tokenLegalCharMap.items():
			# this is a really simplistic approach, but maybe it's ok
			s = s.replace(tokenType.head, replaceChars[0])
			s = s.replace(tokenType.tail, replaceChars[1])
		splitStr = re.findall(reSearchMasterPattern, s, )
		# now iterate over all of the split strings, and replace any that match
		for segment in splitStr:
			newSegment = segment
			for tokenType, replaceChars in self.tokenLegalCharMap.items():
				if not (newSegment.startswith(replaceChars[0]) or newSegment.endswith(replaceChars[1])):
					continue
				newHead = MASTER_GLOBALS_EVALUATOR_KEY + ".resolveToken(_" + tokenType.__name__ + "("
				newTail = f"))"

				newSegment = newSegment.replace(replaceChars[0], newHead)
				newSegment = newSegment.replace(replaceChars[1], newTail)
				if tokenType.tail == nullExpChar:
					newSegment += newTail
				s = s.replace(segment, newSegment)
				break

		return s


class SyntaxNameToStrPass(CustomSyntaxPass):
	"""transformer to replace names with their string values
	NB unchecked, this will convert EVERYTHING, often resulting
	in illegal python code -
	use blacklist and globals diligently
	"""

	def __init__(self, fallbackOnly=True, blacklist:tuple[str]=()):
		"""if fallbackOnly, only replace names that are not already defined in globals
		any names in blacklist will not be replaced"""
		super(SyntaxNameToStrPass, self).__init__()
		self.fallbackOnly = fallbackOnly
		self.blacklist = blacklist

	def _getSyntaxLocalMap(self)->dict:
		"""return dict of {name : value} to update expression globals"""
		return {}

	def visit_Name(self, node: T.Name) -> T.Any:
		"""would this also trip function calls?"""
		if node.id in self.blacklist:
			return node
		if self.fallbackOnly and node.id in self.currentExpGlobals:
			return node
		return ast.Str(s=node.id)


class SyntaxResolveConstantPass(CustomSyntaxPass):
	"""turns any "constant" in expression into
	__evaluator__.resolveConstant("constant")

	figured out that 'Constant' has only recently become the catch-all
	node for constant statements - for 3.7 we have to build in
	a bit of boilerplate
	"""

	def _getCallNodeTree(self, constant:str) -> ast.Call:
		"""return ast.Call node to resolve constant"""
		return ast.Call(
			func=ast.Attribute(
				value=ast.Name(id=MASTER_GLOBALS_EVALUATOR_KEY, ctx=ast.Load()),
				attr="resolveConstant",
				ctx=ast.Load()
			),
			args=[ast.Str(s=constant)],
			keywords=[]
		)

	def visit_Constant(self, node: ast.Constant) -> T.Any:
		return self._getCallNodeTree(node.value)

	def visit_NameConstant(self, node: ast.Constant) -> T.Any:
		return self._getCallNodeTree(node.value)

	def visit_Str(self, node: ast.Str) -> T.Any:
		return self._getCallNodeTree(node.s)

	def visit_JoinedStr(self, node: ast.JoinedStr) -> T.Any:
		return self._getCallNodeTree(node.s)

	def visit_Num(self, node: ast.Num) -> T.Any:
		return self._getCallNodeTree(node.n)

	def visit_Bytes(self, node: ast.Bytes) -> T.Any:
		return self._getCallNodeTree(str(node.s))


class SyntaxFallbackStringPass(CustomSyntaxPass):
	"""if an expression can't be evaluated,
	convert the whole thing to a string
	a very very overkill try / except"""

	def preProcessRawString(self, s:str) ->str:
		try:
			ast.parse(s)
			return s
		except SyntaxError:
			return "\'" + s + "\'"

class SyntaxShorthandFunctionPass(CustomSyntaxPass):
	"""
	support for defining multi-line functions in expressions of form

	():
		<code>


	" 'abc' " - static string
	" (): 'a' + 'b' + 'c' " - function
	" ():
		gg = 'a' + 'b' + 'c'\n
	    return gg + 'wooo' "
	    - multiline function

	in future, can look into type annotations:
	" ()->str: 'a' + 'b' + 'c' "

	or pulling from a global namespace by args and kwargs:
	" ( $time, asset=$currentAsset )->str: 'a' + 'b' + 'c' + asset.name() + time() "


	"""

	@classmethod
	def checkSyntax(cls, text:str):
		"""check syntax of text
		:raises SyntaxError: if syntax is invalid"""
		# check for unbalanced brackets
		contents = syntax.getBracketContents(text)
		flatContents = sequence.flatten(contents)
		if any("(" in i for i in flatContents) or any(")" in i for i in flatContents):
			raise SyntaxError(f"unbalanced brackets: {text}")


	def _isMultiLineFunction(self) ->bool:
		"""check if function is multiline"""
		return "\n" in self.getText().lstrip().rstrip()

	def _expressionTextToPyFunctionText(self, text:str, fnName="") ->str:
		"""convert expression text to python function text"""

		fnName = fnName or self.functionName()

		# check that 'return' keyword is present in body
		signature, body = _splitTextBodySignature(text)

		if not "\n" in body: # single line
			if not body.startswith("return"):
				body = f"return {body}"

		# signature
		if not signature.startswith("def"):
			signature = f"def {fnName}({signature})"

		return f"{signature}:\n\t{body}"

class SyntaxEnsureLambdaPass(CustomSyntaxPass):

	def preProcessRawString(self, s:str) ->str:
		"""check that string starts with
		'lambda' keyword, and add if not"""
		if s.startswith("lambda :"):
			return s
		return "lambda :" + s

@dataclass
class ExpSyntaxProcessor:
	"""holds lists of passes to run over expressions -
	rules that define syntax for a certain context of expression should be
	constant, so reuse these objects where possible.

	A single SyntaxPass object may appear in both string and AST
	lists - this is just to be totally explicit on what runs when.
	"""
	syntaxStringPasses:list[CustomSyntaxPass]
	syntaxAstPasses:list[CustomSyntaxPass]


	def parseRawExpString(self, s:str) ->str:
		"""process string expression -
		first raw string, then parse to AST,
		then visit AST
		"""
		for syntaxPass in self.syntaxStringPasses:
			s = syntaxPass.preProcessRawString(s)
		return s

	@classmethod
	def parseStringToAST(cls, s:str) ->ast.Module:
		"""parse string to AST-
		run this after processing raw string,
		and before AST transform passes"""
		# parse to AST
		inputASTModule = parseStrToASTModule(s)
		#inputASTExt = astModuleToExpression(inputASTModule)
		return inputASTModule

	def processAST(self, expAst:ast.AST) ->ast.AST:
		"""Run AST transform passes"""

		# process AST
		for syntaxPass in self.syntaxAstPasses:
			expAst = syntaxPass.visit(expAst)

		# I think you only need to run this once
		ast.fix_missing_locations(expAst)
		return expAst



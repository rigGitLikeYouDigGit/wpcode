
from __future__ import annotations
import typing as T

import ast, re, pprint
from dataclasses import dataclass
from types import ModuleType, FunctionType
from collections import namedtuple

from wplib import sequence
from wplib.astlib import parseStrToASTModule
from wplib.object import TypeNamespace
from wpexp.constant import MASTER_GLOBALS_EVALUATOR_KEY, EXP_LAMBDA_NAME
from wpexp.error import CompilationError, EvaluationError, ExpSyntaxError
from wpexp.parse import getExpParseFrames, getBracketContents

if T.TYPE_CHECKING:
	pass

"""functions checking expression syntax for user input

a pass may split string into multiple parts, each to be parsed recursively

THIS IS NOT PERFECT YET, and probably never will be.
Nested strings, brackets in strings, expressions constructing other expressions, etc -
these are all allowed conceptually, but the parsing systems here will
not support them.
"""


flatStr = "a + b + c"
testStr = """(): ('a' + ( 'b(' + <time>) + b  )"""
testQuoteStr = """(): ('a' + ('b(' + <time>) ) + "'" + '''"''')"""

"""
compile as 
def _expLambda():
	return ("a" + ("b" + __exp__.eval(
		__exp__.resolveToken("<time>")
		)))
"""

EXP_ACCESS_TOKEN = "@EXP" # token to access expression object, and from there evaluation
# replaced with __exp__ in final expression

#invalidStr = """(): ('a' + (('b' + 'c')) + """

#ExpParseFrame = namedtuple("expFrame", ["startChar", "contents", "endChar"])

# each frame list has opening and closing characters as first and last entries
# and the contents in between
#TODO: optimise if needed

#pprint.pp(getExpParseFrames(testStr), indent=4, depth=10)
#pprint.pp(getExpParseFrames(testStr[:-1]), indent=4, depth=10)

#print(getBracketContents(testStr))
#print(getBracketContents(invalidStr))


# framework for declaring sequence of passes run over expression string -
# this seems like a good balance between efficiency and flexibility.
# I've been struggling with this system for so long, getting something
# on the board is good enough for now

def visitStringParsedFrames(parsedFrames: list[list[str]],
                            transformFn=lambda x : x) -> list[list[str]]:
	"""visit each frame in parsedFrames with transformFn.
	Works from leaves up"""
	for i, s in tuple(enumerate(parsedFrames)):
		if isinstance(s, list):
			parsedFrames[i] = visitStringParsedFrames(s, transformFn)
	parsedFrames = transformFn(parsedFrames)
	return parsedFrames


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


class SyntaxPasses(TypeNamespace):
	"""namespace for syntax passes -
	this was the easiest way to organise them all.

	Extend this to define more passes for specific purposes.
	"""

	class _Base(ast.NodeTransformer, TypeNamespace.base()):
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


		Passes also need to contribute their own separator characters
		for the intermediate frame parsing stage - for now keep it simple

		"""

		@classmethod
		def expressionStringRef(cls)->str:
			"""return a string that will look up the current Expression
			object when evaluated in the expression string's globals"""
			return "__exp__"

		@classmethod
		def evaluatorStringRef(cls)->str:
			return f"{cls.expressionStringRef()}.policy.evaluator"

		def preProcessRawString(self, s: str) -> str:
			"""run any operations on the raw expression string before parsing to ast
			like converting illegal characters and tokens
			override here"""
			return s

		visitStringParsedFrames = visitStringParsedFrames

		def preProcessFrameParsedString(self, stringFrames: list[list[str]]) -> list[list[str]]:
			"""run any operations on the list of string frames,
			intermediate parsed state.
			This will be rejoined into a string before parsing to ast.
			Roll your own visiting logic here for now."""
			return stringFrames


	class KeywordFunctionPass(_Base):
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


	class CharReplacerPass(_Base):
		def __init__(self, charMap:dict[str, str]):
			super().__init__()
			self.charMap = charMap

		def preProcessRawString(self, s: str) -> str:
			"""run any operations on the raw expression string before parsing to ast
			like converting illegal characters and tokens
			override here"""
			for k, v in self.charMap.items():
				s = s.replace(k, v)
			return s


	class TokenReplacerPass(_Base):
		"""transformer to replace custom tokens with functions to resolve them
		"""

		splitChar = "_eyye_"

		def __init__(self,
		             tokenTypes=(ExpTokens.At,
		                         ExpTokens.Dollar,
		                         ExpTokens.PointyBracket)
					 ):
			super().__init__()
			self.tokenTypes = tokenTypes
			self.tokenBaseCharMap = {}
			self.tokenLegalCharMap = {}

		def getSyntaxLocalMap(self)->dict:
			"""return dict of {name : value} to update expression globals"""
			return {"_" + i.__name__ : i for i in self.tokenTypes}

		def stringDefinesExpression(self, s:str) ->bool:
			"""return True if string contains any token characters"""
			for tokenType in self.tokenTypes:
				if tokenType.head in s or tokenType.tail in s:
					return True
			return False

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


	class NameToStrPass(_Base):
		"""transformer to replace names with their string values
		NB unchecked, this will convert EVERYTHING, often resulting
		in illegal python code -
		use blacklist and globals diligently
		"""

		def __init__(self, fallbackOnly=True, blacklist:tuple[str]=()):
			"""if fallbackOnly, only replace names that are not already defined in globals
			any names in blacklist will not be replaced"""
			super().__init__()
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


	class ResolveConstantPass(_Base):
		"""turns any "constant" in expression into
		__evaluator__.resolveConstant("constant")

		figured out that 'Constant' has only recently become the catch-all
		node for constant statements - for 3.7 we have to build in
		a bit of boilerplate
		"""

		def _getCallNodeTree(self, constant:str, evaluatorFnName="resolveConstant") -> ast.Call:
			"""return ast.Call node to resolve constant"""
			return ast.Call(
				func=ast.Attribute(
					value=ast.Name(id=MASTER_GLOBALS_EVALUATOR_KEY, ctx=ast.Load()),
					attr=evaluatorFnName,
					ctx=ast.Load()
				),
				args=[ast.Str(s=constant)],
				keywords=[]
			)


		def visit_Constant(self, node: ast.Constant) -> T.Any:
			return self._getCallNodeTree(node.value)

		def visit_NameConstant(self, node: ast.Constant) -> T.Any:
			return self._getCallNodeTree(node.value, "resolveName")

		def visit_Name(self, node: ast.Name) -> T.Any:

			if node.id == EXP_LAMBDA_NAME:
				return node
			return self._getCallNodeTree(node.id, "resolveName")

		def visit_Str(self, node: ast.Str) -> T.Any:
			return self._getCallNodeTree(node.s)

		def visit_JoinedStr(self, node: ast.JoinedStr) -> T.Any:
			return self._getCallNodeTree(node.s)

		def visit_Num(self, node: ast.Num) -> T.Any:
			return self._getCallNodeTree(node.n)

		def visit_Bytes(self, node: ast.Bytes) -> T.Any:
			return self._getCallNodeTree(str(node.s))



	class ShorthandFunctionPass(_Base):
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
			contents = getBracketContents(text)
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

	class EnsureLambdaPass(_Base):

		def preProcessRawString(self, s:str) ->str:
			"""check that string starts with
			'lambda' keyword, and add if not"""
			if s.startswith("lambda *args, **kwargs :"):
				return s
			return EXP_LAMBDA_NAME + "= lambda *args, **kwargs:" + s
			#return "lambda :" + s



@dataclass
class ExpSyntaxProcessor:
	"""holds lists of passes to run over expressions -
	rules that define syntax for a certain context of expression should be
	constant, so reuse these objects where possible.

	A single SyntaxPass object may appear in both string and AST
	lists - this is just to be totally explicit on what runs when.

	DefinesExpressionFn is a function that takes a string and returns
	true if it is a valid expression for this context.

	This seems more controllable, but more prone to user error than
	delegating to individual SyntaxPass objects.

	"""
	syntaxStringPasses:list[SyntaxPasses.T()]
	syntaxAstPasses:list[SyntaxPasses.T()]
	stringIsExpressionFn:T.Callable[[str], bool] = lambda s: False
	stringIsExpFunctionDefinitionFn:T.Callable[[str], bool] = lambda s: False


	def parseRawExpString(self, s:str) ->str:
		"""process string expression -
		first raw string, then parse to AST,
		then visit AST
		"""
		for syntaxPass in self.syntaxStringPasses:
			s = syntaxPass.preProcessRawString(s)
		return s

	defaultModule = ModuleType("expModule") # copy over all baseline globals keys from this


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

	def compileFinalASTToFunction(
			self,
			finalAst:ast.AST,
			expGlobals:dict)->FunctionType:
		"""compile final ast to function
		this should live somewhere else"""
		#print("compile ast", ast.dump(finalAst))

		c = compile(finalAst, "<expCompile>", "exec", )

		expDict = dict(self.defaultModule.__dict__)
		# update exp globals dict here
		expDict.update(expGlobals)

		exec(c, expDict
		     )
		#print("post exec", expDict.keys())
		return expDict[EXP_LAMBDA_NAME]




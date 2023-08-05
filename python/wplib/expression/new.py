from __future__ import annotations

import wplib.ast
import wplib.expression.evaluator
import wplib.expression.syntax
from wplib.function import addToFunctionGlobals

"""core of refgraph - stores either static value, or function
to be evaluated"""
import typing as T
import ast, pprint
from types import FunctionType, ModuleType
from dataclasses import dataclass

from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import ast as wpast
from wplib.object.serialisable import Serialisable, compileFunctionFromString, serialise
from wplib.object import Visitor
from wplib.object.visit import Visitable, getVisitDestinations, Visitor, recursiveVisitCopy

from wplib.expression import syntax, function as libfn, constant as expconstant
from wplib.expression.error import CompilationError
from wplib.expression.evaluator import ExpEvaluator
from wplib.expression.syntax import ExpSyntaxError, ExpSyntaxProcessor,CustomSyntaxPass

if T.TYPE_CHECKING:
	pass



@dataclass
class ExpressionScope:
	"""object collecting variables accessible within the scope of
	and evaluated expression.

	instance of this object will be added to globals as __scope__.

	Direct access to it is not recommended, write helper functions or
	syntax passes instead.

	I think this is just duplication of evaluator.


	"""
	exp : Expression = None
	# node : ChimaeraNode = None
	# graph : ChimaeraGraph = None

	token = "__scope__"


@dataclass
class ExpPolicy:
	"""general shared object defining behaviour of expression
	"""

	# object to resolve tokens, manage syntax passes, etc
	syntaxProcessor : ExpSyntaxProcessor = ExpSyntaxProcessor([], [])

	# object to evaluate expression in wider context
	# use list of these too to add functionality for different tokens
	evaluator : ExpEvaluator = None
#


class ExpVisitDict(T.TypedDict):
	"""status dict passed into visit functions -
	no way to read back the isStatic flag yet"""
	parentExp : Expression
	isStatic : bool


def transformParseExpStructure(
		obj:object,
		visitData:ExpVisitDict,
)->(object, Expression):
	"""Given any random object, check if it contains any latent expression
	syntax. If so, parse it to a lambda and embed it in structure.

	This does mean we need a second recursive pass to evaluate the full expression -
	could we add "expression stream" objects to a register and only evaluate
	them when requested?
	Cleaner to just return a new copy.

	If not, mark it as static.

	In both cases return a deep copy.

	Inherit expression settings from parent

	Output false if no transform done, true if transform done
	"""
	parentExp = visitData["parentExp"]

	# if obj is string, check if it contains expression syntax
	if isinstance(obj, str):
		if not parentExp.policy.syntaxProcessor.stringDefinesExpressionFn(obj):
			return obj
		processor = parentExp.policy.syntaxProcessor
		# parse string to frozen lambda
		parsedStr = processor.parseRawExpString(obj)
		parsedAst = processor.parseStringToAST(parsedStr)
		processedAst = processor.processAST(parsedAst)
		compiledFn = processor.compileFinalASTToFunction(
			processedAst, {})
		# create expression object with function
		"""TODO: duplication here - probably delegate to expression on actual compilation"""
		expression = Expression(compiledFn, parentExp.policy)
		return expression
	return obj


def transformEvaluateExpStructure(
		parsedObj:object,
		visitData:ExpVisitDict,
)->(object, Expression):
	"""if it's an expression, eval the expression
	and visit the result"""

	# if obj is string, check if it contains expression syntax
	if isinstance(parsedObj, Expression):
		# evaluate expression
		result = parsedObj.eval()
		visitData["parentExp"] = parsedObj
		# visit result - excessive, but only way to account
		# for expressions returning other expressions
		parsedResult = transformParseExpStructure(result, visitData)
		#return recursiveVisitEvaluateExpStructure(parsedResult, parsedObj)

		# don't need to manually recurse here to eval - outer visit will
		# run this eval function over parsed result
		return parsedResult
	return parsedObj




VT = T.TypeVar("VT")
class Expression(Serialisable, T.Generic[VT]):
	"""core of refgraph - stores either static value, or function

	expressions may be static, or a function - each may be defined through
	syntax:


	when set with full function text, break down into regions, store separately,
	compile function

	when set directly with function, first serialise, then save and compile


	expression should track its own error state, but may be set for higher systems -
	if an expression function compiles to a node name, but the higher graph
	determines that name invalid, for example

	expression may either be callable or static. static text may still represent literal higher
	object, for example list. any live reference or variable SHOULD be visible in the AST when
	expression is compiled

	later create policy object to pass in with variables, acceptable return types, etc

	don't directly detect expressions in text here, leave that to syntax passes


	expression may be set with constant value, string or object - recursively parse input
	to detect expression strings.

	Try to return a lambda with frozen function calls to evaluate embedded expressions -
	if an expression returns a token or expression?
	recursively evaluate and visit result of function calls again.



	"""

	# hack here to allow anonymous exp syntax to know the scope it's being run in
	currentExpression : Expression = None # current expression being evaluated
	# when expression is set directly with a function, no way to modify its globals

	defaultModule = ModuleType("expModule") # copy over all baseline globals keys from this

	def __init__(self,
	             value:[T.Callable, VT, str]=None,
	             policy:ExpPolicy=None,
	             name="exp",
	             ):
		"""unsure how best to pass in graph stuff
		for now default to singleton if not specified"""

		self.policy = policy or ExpPolicy()

		self.name = name # nice name for identifying source of expressions, do not rely on this

		self._rawValue = "" # text of expression
		self._processedText = "" # text of expression after syntax passes
		self._parsedAST : ast.AST = None # ast tree of parsed syntax text
		self._finalAst : ast.AST = None # ast tree of text after transform passes
		self._isStatic = False
		self._compiledFn : FunctionType = None

		self.setStructure(value)


	@classmethod
	def _textFromStatic(cls, static:object)->str:
		"""return string text representing static value -
		passing this result to setSourceText() should return an equal object

		very basic for now, augment when everything else is working
		"""
		return str(serialise(static))

	def reset(self):
		self._cachedValue = None
		self._rawValue = "" # text of expression
		self._processedText = "" # text of expression after syntax passes
		self._parsedAST : ast.AST = None # ast tree of parsed syntax text
		self._finalAst : ast.AST = None # ast tree of text after transform passes
		self._isStatic = False
		self._compiledFn : FunctionType = None


	def setStatic(self, value:T.Any):
		"""set expression to a static value -
		any complex functions inside are evaluated and result stored.

		to get live value, set expression as lambda
		"""
		self._cachedValue = value
		self._isStatic = True
		self._rawValue = self._textFromStatic(value)
		try:
			self._finalAst = wplib.ast.parseStrToASTExpression(self._rawValue)
		except SyntaxError:
			self._finalAst = None
		self._compiledFn = None # not sure if this should be compiled ast expression



	def _compileFinalASTToFunction(
			self,
			finalAst:ast.Module,
			expGlobals:dict)->FunctionType:
		"""compile final ast to function"""
		c = compile(finalAst, "<expCompile>", "exec")

		expDict = dict(self.defaultModule.__dict__)
		# update exp globals dict here
		expDict.update(expGlobals)

		exec(c, expDict
		     )
		return expDict[expconstant.EXP_LAMBDA_NAME]


	def setSourceText(self, text:str):
		"""set expression to text
		-process text
		-parse text to ast
		-process ast
		-compile ast to function

		assume that all expressions hold a function, even if it's just a static value

		retrieving a full function rather than just running 'eval' all the time
		costs more in compilation, but it's way faster once compiled

		maybe there's some world where we use eval first and compile in the background

		"""
		self.reset()
		self._rawValue = text
		try:
			self._processedText = self.policy.syntaxProcessor.parseRawExpString(text)
		except ExpSyntaxError as e:
			raise e

		self._parsedAST = self.policy.syntaxProcessor.parseStringToAST(self._processedText)

		try:
			self._finalAst = self.policy.syntaxProcessor.processAST(self._parsedAST)
		except Exception as e:
			raise e

		print("parsedAST")
		print(self._parsedAST)
		print(ast.dump(self._finalAst))
		#self._compiledFn = wpast.compileASTExpression(self._finalAst)
		self._compiledFn = self._compileFinalASTToFunction(
			self._finalAst, {})
		print("compiledFn")
		print(self._compiledFn, self._compiledFn())



	def setFunction(self, fn:FunctionType):
		"""set expression to callable function"""
		self._cachedValue = None
		self._compiledFn = fn
		self._isStatic = False
		#self._text = serialiseFunction(fn)["code"]
		self._rawValue = libfn.getFnSource(fn)
		self._finalAst = wplib.ast.parseStrToASTExpression(self._rawValue)

	def setStructure(self, structure:object):
		"""set expression to structure - recursively parse structure to find any embedded functions
		"""
		self._cachedValue = None
		self._compiledFn = None
		self._rawValue = structure
		self._parsedStructure = recursiveVisitCopy(
			structure, transformParseExpStructure,
			ExpVisitDict(parentExp=self, isStatic=False)
		                                           )


	def eval(self):
		"""evaluate expression and return result -
		update globals with references to evaluator and self.

		There are edge cases here where an inner expression may update globals,
		then remove the keys,
		in the midst of an outer expression trying to run
		"""
		oldGlobals = dict(globals())
		expGlobalsMap = {
			# __exp__
			expconstant.MASTER_GLOBALS_EXP_KEY : self,
			# __evaluator__
			expconstant.MASTER_GLOBALS_EVALUATOR_KEY : self.policy.evaluator,
		}
		globals().update(expGlobalsMap)

		# recursively evaluate any sub-expressions
		result = recursiveVisitCopy(
			self._parsedStructure, transformEvaluateExpStructure,
			ExpVisitDict(parentExp=self, isStatic=False)
		)

		# restore globals
		for k in expGlobalsMap:
			globals().pop(k)

		# return result
		return result


	def functionName(self) ->str:
		"""get function name"""
		return self.name + "_fn"


	def _getExpVars(self) ->dict:
		"""get global variables for expression"""
		scope = self.policy.getScopeFn()
		scope.exp = self
		return {
			"__scope__": scope
		}


	def getText(self) ->str:
		"""get text representation of expression
		if function, return decompiled function text"""
		return self._rawValue

	def _compute(self):
		"""compute expression value and store in cache"""
		if self.isStatic():
			#self._cachedValue = eval(self._ast)
			pass # value already cached when set
		else:
			print("updating with", self._getExpVars())
			print("base globals", *self._compiledFn.__globals__.keys())

			# directly setting function globals is apparently illegal

			# self._compiledFn.__globals__ = {
			# 	**self._compiledFn.__globals__,
			# 	**self._getExpScope()
			# }

			addToFunctionGlobals(self._compiledFn, self._getExpVars())

			self._cachedValue = self._compiledFn()
		return self._cachedValue

	def __hash__(self):
		"""for now removing hashes from expression results -
		using as dict has to be done through explicit objects"""
		return id(self)

	def __str__(self):
		"""for str, return the text of this expression"""
		return f"<{self.__class__.__name__} {self.name} : {self.getText()}>"

	def __repr__(self):
		return str(self)

	def copy(self) ->Expression:
		"""copy expression"""
		newExp = Expression(name=self.name, graph=self.dGraph)
		newExp._rawValue = self._rawValue
		newExp._cachedValue = self._cachedValue
		newExp._compiledFn = self._compiledFn
		newExp._isStatic = self._isStatic
		newExp._finalAst = self._finalAst
		return newExp


	def serialise(self) ->dict:
		"""serialise expression"""

		return {
			"name": self.name,
			"text": self.getText(),
		}

	@classmethod
	def deserialise(cls, data:dict):
		exp = cls(name=data["name"])
		exp.set(data["text"])

	def printRaw(self):
		"""print raw expression text"""
		pprint.pprint(self._rawValue)

	def printParsed(self):
		"""print parsed expression"""
		pprint.pprint(self._parsedStructure)

class ExpTools:
	"""namespace for collecting a load of expression tools"""
	ExpressionPolicy = ExpPolicy
	ExpressionEvaluator = ExpEvaluator
	#class SyntaxPasses:


if __name__ == '__main__':
	# print(getBracketContents(testStr))
	# print(restoreBracketContents(getBracketContents(testStr)))

	pass

	#exp = Expression()
	#print(exp)
	#exp.setSourceText("a + b + c")
	#exp.setSourceText(expconstant.EXP_LAMBDA_NAME + " = lambda : 'a' + 'b' ")

	def embedFn():
		"""embed function in expression"""
		print("embed hello")

	def recursiveFn():
		print("recursive hello")
		return "$recursiveExp"

	expStructure = [
		("a", "$time"),
		2,
		{
			"b": recursiveFn,
			"c": "(): print('hello')",
			"$asset": False
		},

		("d", embedFn),
	]

	tokenRule = Sy

	processor = ExpSyntaxProcessor()

	exp = Expression()

	exp.setStructure(expStructure)

	exp.printRaw()
	exp.printParsed()





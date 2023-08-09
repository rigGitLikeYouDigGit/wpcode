from __future__ import annotations

import wplib.ast
import wplib.expression.evaluator
import wplib.expression.syntax
from wplib.function import addToFunctionGlobals

"""core of refgraph - stores either static value, or function
to be evaluated"""
import typing as T
import ast, pprint, re
from types import FunctionType, ModuleType
from dataclasses import dataclass

from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import ast as wpast
from wplib.object.serialisable import Serialisable, compileFunctionFromString, serialise
from wplib.object import Visitor
from wplib.object.visit import Visitable, getVisitDestinations, Visitor, recursiveVisitCopy

from wplib.expression import syntax, function as libfn, constant as expconstant
from wplib.expression.error import CompilationError, EvaluationError, ExpSyntaxError
from wplib.expression.evaluator import ExpEvaluator
from wplib.expression.syntax import ExpSyntaxProcessor, SyntaxPasses


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

	def expGlobalMap(self)->dict:
		"""map of globals to update expression with"""
		return {
			expconstant.MASTER_GLOBALS_EVALUATOR_KEY : self.evaluator,
		}
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
	#print("transformParseExpStructure", obj)
	if isinstance(obj, str):
		#print("defines", parentExp.policy.syntaxProcessor.stringDefinesExpressionFn(obj))
		if not parentExp.policy.syntaxProcessor.stringDefinesExpressionFn(obj):
			return obj
		processor = parentExp.policy.syntaxProcessor
		# parse string to frozen lambda
		parsedStr = processor.parseRawExpString(obj)
		parsedAst = processor.parseStringToAST(parsedStr)
		processedAst = processor.processAST(parsedAst)
		# print("processed AST:")
		# print(ast.dump(processedAst))
		compiledFn = processor.compileFinalASTToFunction(
			processedAst, parentExp.getExpGlobals())
		# create expression object with function
		return compiledFn
		"""TODO: duplication here - probably delegate to expression on actual compilation"""
		# expression = Expression(compiledFn, parentExp.policy)
		# return expression
	return obj


def transformEvaluateExpStructure(
		parsedObj:object,
		visitData:ExpVisitDict,
)->(object, Expression):
	"""if it's an expression, eval the expression
	and visit the result"""
	#print("transformEvaluateExpStructure", parsedObj, type(parsedObj), callable(parsedObj))
	if callable(parsedObj):
		return parsedObj()

	# if obj is string, check if it contains expression syntax
	# if isinstance(parsedObj, Expression):
	# 	# evaluate expression
	# 	result = parsedObj.eval()
	# 	visitData["parentExp"] = parsedObj
	# 	# visit result - excessive, but only way to account
	# 	# for expressions returning other expressions
	# 	parsedResult = transformParseExpStructure(result, visitData)
	# 	#return recursiveVisitEvaluateExpStructure(parsedResult, parsedObj)
	#
	# 	# don't need to manually recurse here to eval - outer visit will
	# 	# run this eval function over parsed result
	# 	return parsedResult
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

	def getExpGlobals(self)->dict:
		"""return dict of globals to pass to expression"""
		globalsMap = self.policy.expGlobalMap()
		# add in reference to self
		globalsMap[expconstant.MASTER_GLOBALS_EXP_KEY] = self
		return globalsMap

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
		parsed = recursiveVisitCopy(
			structure, transformParseExpStructure,
			ExpVisitDict(parentExp=self, isStatic=False)
		                                           )
		#print("parsed result", parsed, type(parsed))
		self._parsedStructure = parsed

	def rawStructure(self)->object:
		"""return raw structure of expression"""
		return self._rawValue

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
		# for k in expGlobalsMap:
		# 	globals().pop(k)

		# return result
		return result

	def resultStructure(self)->object:
		"""return result structure of expression -
		eval if needed;
		if not dirty, return cached value"""
		return self.eval()


	def __call__(self, *args, **kwargs):
		return self.eval()


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
		return f"<Exp {self.name} : {self.getText()}>"

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

	# test maya set expressions
	setExpStr = "root_GRP/mid/* + *_PLY"

	class ElementSet(set):

		def __init__(self, filterStr:str):
			super().__init__()
			self.filterStr = filterStr

		def populate(self):
			"""populate set with elements matching filter string -
			clears set first"""
			raise NotImplementedError



	class MayaSetSyntaxPass(SyntaxPasses._Base):
		"""a star directly adjoined to a string is considered a wildcard match
		a star separated by a space is considered a set intersection
		"""
		STAR_CHAR = "__STX_STAR__"
		DOUBLE_STAR_CHAR = "__STX_DOUBLE_STAR__"
		SLASH_CHAR = "__STX_SLASH__"

		def preProcessRawString(self, s: str) -> str:

			for char, replace in [
				("\*", self.STAR_CHAR),
				("\*\*", self.DOUBLE_STAR_CHAR),
			]:

				for m in reversed(tuple(re.finditer(char, s))):
					# try:
					# 	if s[m.start() - 1] == " ":
					# 		# is space, skip
					# 		continue
					# except IndexError:
					# 	pass
					s = s[:m.start()] + replace + s[m.end():]

			# replace slashes - only space-separated are retained as divisions
			for m in reversed(tuple(re.finditer("\/", s))):
				if m.start() - 1 != " " and m.end() + 1 != " ":
					s = s[:m.start()] + self.SLASH_CHAR + s[m.end():]

			print("preProcessRawString", s)
			return s

		def visit_Name(self, node: Name) -> Any:
			"""visit name node - any raw names converted to
			maya list operations that match that string"""

			restoreId = node.id.replace(self.STAR_CHAR, "*").replace(
				self.DOUBLE_STAR_CHAR, "**").replace(self.SLASH_CHAR, "/")

			node.id = restoreId
			return node


	mayaPass = MayaSetSyntaxPass()
	resolveNamePass = SyntaxPasses.ResolveConstantPass()
	ensureLambdaPass = SyntaxPasses.EnsureLambdaPass()

	mayaSetSyntaxProcessor = ExpSyntaxProcessor(
		[mayaPass, ensureLambdaPass
		 ],
		syntaxAstPasses=[mayaPass, resolveNamePass],
		stringDefinesExpressionFn=lambda s : True
	)

	class MayaSetEvaluator(ExpEvaluator):

		def resolveName(self, name:str):
			print("resolveName", name)
			return name
		pass



		# def resolveConstant(self, constant:(str, int, float, bool, tuple, list, dict)):
		# pass

	mayaEvaluator = MayaSetEvaluator()

	mayaSetPolicy = ExpPolicy(
		mayaSetSyntaxProcessor,
		mayaEvaluator
	)

	exp = Expression(policy=mayaSetPolicy)
	exp.setStructure(setExpStr)
	exp.printParsed()

	print("result", exp, exp.eval())

	#exp = Expression()
	#print(exp)
	#exp.setSourceText("a + b + c")
	#exp.setSourceText(expconstant.EXP_LAMBDA_NAME + " = lambda : 'a' + 'b' ")

	# def embedFn():
	# 	"""embed function in expression"""
	# 	print("embed hello")
	#
	# def recursiveFn():
	# 	print("recursive hello")
	# 	return "$recursiveExp"
	#
	# expStructure = [
	# 	("a", "$time"),
	# 	2,
	# 	{
	# 		"b": recursiveFn,
	# 		"c": "(): print('hello')",
	# 		"$asset": False
	# 	},
	#
	# 	("d", embedFn),
	# ]
	#
	# tokenRule = Sy
	#
	# processor = ExpSyntaxProcessor()
	#
	# exp = Expression()
	#
	# exp.setStructure(expStructure)
	#
	# exp.printRaw()
	# exp.printParsed()





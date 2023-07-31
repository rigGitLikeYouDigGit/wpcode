from __future__ import annotations

import wplib.ast
import wplib.expression.evaluator
import wplib.expression.syntax
from wplib.function import addToFunctionGlobals

"""core of refgraph - stores either static value, or function
to be evaluated"""
import typing as T
import ast
from types import FunctionType
from dataclasses import dataclass

from wplib.object.serialisable import Serialisable, compileFunctionFromString, serialise
from wplib.object import Visitor
from wplib.expression import syntax, function as libfn
from wplib.expression.error import CompilationError
from wplib.expression.evaluator import ExpEvaluator
from wplib.expression.syntax import ExpSyntaxError, ExpSyntaxProcessor,CustomSyntaxPass

if T.TYPE_CHECKING:
	pass

class LiteralVisitor(Visitor):
	"""where possible, converts references to classes and objects into
	eval-able literal strings.

	Not used yet - use this to let complex objects persist in string representation -
	map their representative strings back to real objects"""

	def visit(self, obj, visitData:Visitor.VisitData):
		"""convert object to literal string"""
		if isinstance(obj, type):
			return ".".join((obj.__module__, obj.__name__))
		elif hasattr(obj, "__class__"):
			return ".".join((obj.__class__.__module__, obj.__class__.__name__)) + "()"
		else:
			return repr(obj)




flatStr = "a + b + c"
testStr = """(): ('a' + ('b' + 'c'))"""
invalidStr = """(): ('a' + (('b' + 'c'))"""

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
class ExpressionPolicy:
	"""general shared object defining behaviour of expression
	"""

	syntaxProcessor : ExpSyntaxProcessor
	evaluator : ExpEvaluator = None # object to resolve tokens, manage syntax passes, etc
#


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

	"""

	# hack here to allow anonymous exp syntax to know the scope it's being run in
	currentExpression : Expression = None # current expression being evaluated
	# when expression is set directly with a function, no way to modify its globals


	def __init__(self,
	             value:[T.Callable, VT, str]=None,
	             policy:ExpressionPolicy=None,
	             name="exp",
	             ):
		"""unsure how best to pass in graph stuff
		for now default to singleton if not specified"""

		self.policy = policy or ExpressionPolicy()

		self.name = name # nice name for identifying source of expressions, do not rely on this

		self._cachedValue = None
		self._text = "" # text of expression
		self._ast : ast.AST = None # ast tree of text
		self._isStatic = False
		self._compiledFn : FunctionType = None

		self.setExp(value)

	@classmethod
	def _textFromStatic(cls, static:object)->str:
		"""return string text representing static value -
		passing this result to setSourceText() should return an equal object

		very basic for now, augment when everything else is working
		"""
		return str(serialise(static))


	def setStatic(self, value:T.Any):
		"""set expression to a static value -
		any complex functions inside are evaluated and result stored.

		to get live value, set expression as lambda
		"""
		self._cachedValue = value
		self._isStatic = True
		self._text = self._textFromStatic(value)
		try:
			self._ast = wplib.ast.parseStrToASTExpression(self._text)
		except SyntaxError:
			self._ast = None
		self._compiledFn = None # not sure if this should be compiled ast expression



	def setSourceText(self, text:str):
		"""set expression to text
		First check if text defines a function, or if it contains any dynamic variables -
		if so, run syntax passes and compile function, otherwise set as static value
		"""
		self._cachedValue = None
		self._isStatic = False
		self._text = text
		self._ast = wplib.ast.parseStrToASTExpression(self._text)
		self._compiledFn = None

	def setFunction(self, fn:FunctionType):
		"""set expression to callable function"""
		self._cachedValue = None
		self._compiledFn = fn
		self._isStatic = False
		#self._text = serialiseFunction(fn)["code"]
		self._text = libfn.getFnSource(fn)
		self._ast = wplib.ast.parseStrToASTExpression(self._text)

	def setExp(self, exp:(T.Callable, str, VT)):
		"""set expression
		if function, decompile function text
		if string, check syntax, then check if defines function

		:raises SyntaxError: if syntax is invalid
		:raises CompilationError: if function cannot be compiled
		"""
		# pass in a function and use it directly
		if isinstance(exp, FunctionType):
			self.setFunction(exp)
			return

		# if string is passed, run all syntax passes on it
		elif isinstance(exp, str):


			# if text defines a function, compile it and use it
			if syntax.textDefinesExpFunction(exp):
				# add any missing keywords to get back to normal python syntax
				formattedText = self._expressionTextToPyFunctionText(exp)
				try:
					compiledFn = self._compileFunctionText("\n" + formattedText + "\n")
				except Exception as e:
					raise CompilationError(f"failed to compile function: {formattedText}\n\t from raw: {exp} \n\t{e}")
				self.setFunction(compiledFn)
				return

			# first see if it can be compiled as a literal
			try:
				result = ast.literal_eval(exp)
				self.setStatic(result)
				return
			except (ValueError, SyntaxError):
				pass

		# set expression value to literal string value passed in
		self.setStatic(exp)

	def isStatic(self) ->bool:
		return self._isStatic

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



	def _compileFunctionText(self, text:str) ->T.Callable:
		"""compile function text"""
		fnName = text.strip().split("(")[0].split("def")[1].strip()
		return compileFunctionFromString(text, fnName,
		                                 fnGlobals={"TESTGLOBAL" : 69}
		                                 #fnGlobals=self._getExpScope()
		                                 )

	def _textDefinesLiteral(self, text:str) ->bool:
		"""check if text defines a literal value"""
		return not syntax.textDefinesExpFunction(text)


	def getText(self) ->str:
		"""get text representation of expression
		if function, return decompiled function text"""
		return self._text

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
		newExp._text = self._text
		newExp._cachedValue = self._cachedValue
		newExp._compiledFn = self._compiledFn
		newExp._isStatic = self._isStatic
		newExp._ast = self._ast
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

class ExpTools:
	"""namespace for collecting a load of expression tools"""
	ExpressionPolicy = ExpressionPolicy
	ExpressionEvaluator = ExpEvaluator
	#class SyntaxPasses:


if __name__ == '__main__':
	# print(getBracketContents(testStr))
	# print(restoreBracketContents(getBracketContents(testStr)))

	pass

	# exp = Expression()
	# #print(exp)
	#
	# exp.set(" (): 'a' + 'b' + 'c' ")
	# exp.set(" ( gg:str ): 'a' + 'b' + 'c' ")
	# exp.set(" 'a' + 'b' + 'c' ")



from __future__ import annotations

from wplib.ast import astModuleToExpression, parseStrToASTModule, evalASTExpression
from wplib.expression.constant import MASTER_GLOBALS_EVALUATOR_KEY
#from wplib.expression.syntax import ExpressionToken


class ExpEvaluator:
	"""main class to actually evaluate expressions
	separate syntax passes modify the expression text
	and build dict of globals -

	this object is accessible in final expression as
	__evaluator__

	special "resolve" functions are called on expression
	tokens and constants.

	Subclass this to change behaviour of expression itself, once full
	syntax is known.
	"""


	def __init__(self):
		self.expStrInput = ""
		self.expInputData = None
		self.expGlobals : dict = None

	def resolveToken(self, token: ExpressionToken):
		"""resolve specially marked tokens to their final value"""
		return token.content

	def resolveConstant(self, constant:(str, int, float, bool, tuple, list, dict)):
		"""called on any constants remaining in expression"""
		return constant

	def resolveName(self, name:str):
		"""called on any names remaining in expression"""
		print("resolveName", name)
		return self.resolveConstant(name)

	def evalExpression(self, expInput:str, expGlobals:dict, inputData:dict=None, frameName="<expression>")->any:
		"""evaluate an expression string, using custom syntax
		and resolve functions
		syntax transform passes should have already been run,
		expInput here is final processed, unparsed string
		"""
		self.expStrInput = expInput
		self.expGlobals = dict(expGlobals)
		self.expInputData = inputData

		# add this object to globals as __evaluator__
		self.expGlobals[MASTER_GLOBALS_EVALUATOR_KEY] = self

		# evaluate expression
		parsedExp = astModuleToExpression(parseStrToASTModule(expInput))
		result = evalASTExpression(parsedExp, self.expGlobals, frameName=frameName)
		return result
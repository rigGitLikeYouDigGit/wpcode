from __future__ import annotations
import typing as T

from wplib.ast import astModuleToExpression, parseStrToASTModule, evalASTExpression
from wplib.expression.constant import MASTER_GLOBALS_EVALUATOR_KEY
#from wplib.expression.syntax import ExpressionToken
if T.TYPE_CHECKING:
	from wplib.expression.syntax import SyntaxPasses, ExpTokens

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

	This class does not control overall evaluation of whole expression,
	only the evaluation of individual tokens and constants.
	"""

	def resolveToken(self, token:ExpTokens.T() ):
		"""resolve specially marked tokens to their final value"""
		return token.content

	def resolveConstant(self, constant:(str, int, float, bool, tuple, list, dict)):
		"""called on any constants remaining in expression"""
		return constant

	def resolveName(self, name:str):
		"""called on any names remaining in expression"""
		#print("resolveName", name)
		return self.resolveConstant(name)

	def __getattr__(self, item):
		"""resolve any other attributes to constants"""
		return self.resolveName(item)

	# def __call__(self, *args, **kwargs):
	# 	"""resolve a longer string"""
	# 	return self.resolveConstant(args)


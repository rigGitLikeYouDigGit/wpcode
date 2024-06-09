
from __future__ import annotations

from wplib.expression.evaluator import ExpEvaluator
from wpexp.syntax import CustomSyntaxPass, SyntaxKeywordFunctionPass, ExpressionToken, AtToken, DollarToken, \
	PointyToken, SyntaxTokenReplacerPass, SyntaxNameToStrPass, SyntaxResolveConstantPass, SyntaxFallbackStringPass

"""now is time, my friend

libs and helpers for processing string expressions and ASTs

for large and complex systems, I often find myself quite lost when
tackling them - I am but a poor rigging boy, after all.
sometimes when I'm scolded for making things complicated, it's
justified - often though, it comes of me feeling my way forwards
in the dark, in small steps from ground level,
trying to find a way forwards that doesn't seem wrong.
Rarely, I happen on a way that seems right - in general though, as
long as I keep to what isn't obviously a bad idea,
things tend to work out  

desperately trying to keep this small enough to not warrant
importing lark

we defer as much actual evaluation as we can to python itself -
we shouldn't need to run multiple syntax passes.
don't try and resolve an expression token to another token.
please.
"""
#import parser
from dataclasses import dataclass
import typing as T
try:
	from wplib.sequence import flatten
except:
	pass


# compatibility with pre-3.8 python
try:
	from ast import unparse
except ImportError:
	from astunparse import unparse





# convertFnAstCode = ast.parse("__in__fn(a, b)")
# #print(ast.dump(convertFnAstCode, indent=2))


# converting between strings and ast - pass ast results to converters
# should be moved to normal lib file, but no point yet

# specific combined expressions from strings, in case performance warrants

# key in expression globals pointing back to master evaluator object


# super inefficient to pre-process, parse, process, unparse, and post-process
# I don't think we lose much if we run ALL pre-processes, parse once, ALL processes.


# substitute language keywords for equivalent magic methods -
# keywords take priority over the normal processing if they're found, so
# need to remove them first


# pattern to split a string based on whitespace, brackets, operators etc

testTokenExp = "@P + <(eyyy)> + $P"
#result = SyntaxTokenReplacerPass().processExpression(testTokenExp, {}, returnAST=False)
# seems to work


testNameExp = """a + "b" + c + d.gg + aw("hell") """
#result = SyntaxNameToStrPass().processExpression(testNameExp, {}, returnAST=False)

testCallExp = """__evaluator__.resolveConstant("add")"""
#print(ast.dump(ast.parse(testCallExp), indent=2))
#raise


@dataclass
class Expression:
	"""master catch-all for single import and assignment"""
	syntaxPasses : T.Tuple[CustomSyntaxPass, ...] = () # ordered syntax passes
	evaluator : ExpEvaluator = ExpEvaluator()

	def eval(self, expStr:str, expGlobals:dict=None, inputData=None, frameName="<expression>"):
		"""main entrypoint for evaluating an expression,
		runs syntax passes, building exp globals, and then
		calls evaluator to evaluate expression
		"""
		if expGlobals is None:
			expGlobals = {}
		print("start eval", expStr, "with data", inputData)
		for syntaxPass in self.syntaxPasses:
			expStr = syntaxPass.processExpression(expStr, expGlobals, returnAST=False)
			print("exp:", expStr)
		return self.evaluator.evalExpression(expStr, expGlobals, inputData, frameName=frameName)


class ExpressionTools:
	"""simple namespace to avoid 20 lines of imports when
	working with expressions"""
	class Tokens:
		Base = ExpressionToken
		DollarToken = DollarToken
		PointyToken = PointyToken
		AtToken = AtToken

	class SyntaxPasses:
		Base = CustomSyntaxPass
		SyntaxResolveConstantPass = SyntaxResolveConstantPass
		SyntaxTokenReplacerPass = SyntaxTokenReplacerPass
		SyntaxNameToStrPass = SyntaxNameToStrPass
		SyntaxKeywordFunctionPass = SyntaxKeywordFunctionPass
		SyntaxFallbackStringPass = SyntaxFallbackStringPass

	ExpressionEvaluator : ExpEvaluator = ExpEvaluator



if __name__ == '__main__':

	testExp = "2 * 3"
	testExp = "'*'"
	#testExp = "a or b"
	#testExp = "2 or 3"

	#print("result", result)

	# exp = Expression(
	# 	(SyntaxKeywordFunctionPass(),
	# 	 SyntaxNameToStrPass(),
	# 	 SyntaxResolveConstantPass(),
	# 	 )
	# )
	# exp.eval(testExp)



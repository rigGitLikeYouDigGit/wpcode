from __future__ import annotations

import ast
from types import CodeType

"""have to name this module 'astlib' because 'ast' is already taken by the stdlib"""

def parseStrToASTModule(s:str)->ast.Module:
	"""ast has some tricky stuff around parsing vs compiling vs eval-ing,
	all of which can result in Expression or Module AST objects"""
	baseTree = ast.parse(s,  # mode="eval"
						 )
	return baseTree


def astModuleToExpression(m:ast.Module)->ast.Expression:
	"""counterintuitive, not documented anywhere as far as I can tell,
	but this works consistently now
	"""
	return ast.Expression(m.body[0].value)


def evalASTExpression(e:ast.Expression, expGlobals:dict, frameName="<expression>"):
	"""evaluates an ast expression, then returns its result"""
	assert isinstance(e, ast.Expression), "input must be an Expr, this is super finnicky"
	return eval(compile(e, frameName, "eval"),
		{}, expGlobals)


def compileASTExpressionFromString(expStr:str, frameName="<expression>")->CodeType:
	"""compile an expression string to an ast expression"""
	return compile(
		ast.Expression(ast.parse(expStr).body[0].value),
		frameName,
		"eval"
	)


def parseStrToASTExpression(s:str)->ast.Expression:
	"""parse a string to an ast expression, can be passed to syntax rules"""
	return ast.Expression(ast.parse(s).body[0].value)


def compileASTExpression(e:ast.Expression, frameName="<expression>")->CodeType:
	"""compile an ast expression to a code object.
	Run this after AST has been fully processed, more performant if it's compiled
	"""
	return compile(e, frameName, "eval")
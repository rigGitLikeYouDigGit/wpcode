
from __future__ import annotations

import ast
from types import FunctionType, LambdaType
import inspect

from wplib.object.serialisable import isLambda, compileFunctionFromString

"""
extracting text from functions, loading functions from text

for any function to be valid here, there MUST be literal
source text somewhere in the program.
if loaded from a text file, we preserve the source text as a 
dynamic attribute __sourcetext__ on the function"""


listLambda = lambda x: isinstance(x, list)
multiLineLambda = (
	lambda x: isinstance(x, list) and \
		any(isinstance(y, list) for y in x))

# a compound lambda WOULD be extremely thorny, but
# the whole point of a lambda prevents access to internal
compoundLambda = lambda x : "gg" == (lambda y: y)
# we don't need to solve this

def firstUnbalancedIndex(s):
	"""return the index of the first unbalanced closing bracket in string
	this doesn't account for bracket characters in strings"""
	pairs = {"{": "}", "(": ")", "[": "]"}
	closing = pairs.values()
	stack = []
	for i, c in enumerate(s):
		if c in "{[(":
			stack.append(c)
		elif stack and c == pairs[stack[-1]]:
			stack.pop()
		elif c in closing:
			return i
	return -1



def _getLambdaSource(fn:LambdaType)->str:
	"""return the source code for a lambda function as a string,
	very simple source filtering - discard everything before the first
	'lambda' keyword and everything after the last ')'"""
	raw = inspect.getsource(fn)
	# find first lambda keyword
	lambdaIndex = raw.find("lambda")
	# remove anything preceding it
	raw = raw[lambdaIndex:]

	# check there are no trailing brackets from formatting
	trimIndex = firstUnbalancedIndex(raw)
	raw = raw[:trimIndex]

	return raw

def _getDefSource(fn:FunctionType)->str:
	"""return the source code for a function as a string,
	additional issues here in case outer variables have been included,
	but those are left to higher systems

	only concerned with literal source code text
	"""
	raw = inspect.getsource(fn)

	# remove any indents affecting whole function
	lines = raw.split("\n")
	indent = len(lines[0]) - len(lines[0].lstrip())
	raw = "\n".join(line[indent:] for line in lines)

	return raw

def getFnSource(fn:(LambdaType, FunctionType))->str:
	"""return the source code for a function as a list of lines"""
	if hasattr(fn, "__sourcetext__"):
		return fn.__sourcetext__
	if isLambda(fn):
		return _getLambdaSource(fn)
	return _getDefSource(fn)

def loadFnFromSource(source:str, fnName:str="")->FunctionType:
	"""load a function from source code"""
	fn = compileFunctionFromString(source, fnName)
	fn.__sourcetext__ = source
	return fn

def ensureASTIsFunction(ast:ast.AST)->ast.Lambda:
	"""ensure that an ast is a function, if it is a string, load it as a function"""
	if isinstance(ast, str):
		return loadFnFromSource(ast)
	return ast

if __name__ == '__main__':

	def defFn():
		return "test"

	print(isLambda(listLambda))
	print(getFnSource(listLambda))

	print(getFnSource(multiLineLambda))

	print(getFnSource(defFn))






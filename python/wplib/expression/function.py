
from __future__ import annotations

import ast
from types import FunctionType, LambdaType
import inspect, tempfile
from collections import namedtuple

#from wplib.object.serialisable import isLambda, compileFunctionFromString

"""
extracting text from functions, loading functions from text

for any function to be valid here, there MUST be literal
source text somewhere in the program.
if loaded from a text file, we preserve the source text as a 
dynamic attribute __sourcetext__ on the function"""


def isLambda(fn:FunctionType)->bool:
	"""check if function is lambda
	seems like once defined, there is literally no difference between lambdas
	and functions in compiled code - relying on name like this isn't a language
	specification, so might break, but it seems the only way for now"""
	return fn.__code__.co_name == "<lambda>"

def serialiseParameter(param:inspect.Parameter)->dict:
	return {"name" : param.name,
	        "default" : param.default,
	        "annotation" : param.annotation,
	        "kind" : param.kind,
	        }

def serialiseSignature(sig:inspect.Signature)->dict:
	return {"parameters" : [serialiseParameter(p) for p in sig.parameters.values()],
	        "return_annotation" : sig.return_annotation,
	        }

def serialiseFunction(fn:FunctionType)->dict:
	"""For now until third-party support for this improves, decompiling
	a compiled function is out.

	stick to getting and processing function source lines.
	"""

	try:
		#from decompyle3 import code_deparse
		from uncompyle6 import code_deparse
	except ImportError:
		raise ImportError("decompyle3 is required for serialising functions")
	signature = inspect.signature(fn)
	sysvertype = namedtuple('sysvertype', 'major minor micro releaselevel serial')
	version38 = sysvertype(3, 8, 0, 'final', 0)
	with open(tempfile.mktemp(), "w") as f:
		deparsed = code_deparse(fn.__code__, version=version38, compile_mode="exec",
		                        # out=sys.stdout,
		                        out=f,
		                        )
	return {"signature" : serialiseSignature(signature),
	        "code" : deparsed.text,
	        "name" : fn.__name__,
	        "isLambda" : isLambda(fn)}

def compileFunctionFromString(fnStr:str, fnName="", isLambda=False,
                              fnGlobals={})->FunctionType:
	"""compile function from string - requires input of
	'def fnName(): return 2 + 3'
	or
	'lambda x: x + 1'
	with all python syntax as normal
	"""

	moduleName = "<compiled function>"
	if isLambda: # convert to 'LAMBDA_FN = lambda x: x + 1'
		moduleName = "<compiled lambda>"
		fnName = 'LAMBDA_FN'
		fnStr = fnName + " = " + fnStr

	c = compile(fnStr, "<string>", "exec")

	print("exec with globals", fnGlobals)
	raise

	# update with custom "__builtins__" key to add custom variables
	globalsDict = builtins.__dict__.copy()
	globalsDict.update(fnGlobals)

	newModule = ModuleType(moduleName)
	exec(c, newModule.__dict__, __globals=fnGlobals)

	if 0: # debug
		print("dynamic module contains:")
		for k, v in newModule.__dict__.items():
			if k == "__builtins__":
				continue
			print(k, v)

	return newModule.__dict__[fnName]

#print(compileFunctionFromString("lambda : 2 + 3", isLambda=True))



def deserialiseFunction(fnData:dict, fnGlobals={})->FunctionType:
	"""deserialise function from dict"""

	if fnData["isLambda"]:
		# remove 'return' from lambda definition
		lambdaCode = fnData["code"].replace("return ", "", 1)
		shellStr = f"""LAMBDA_FN = lambda {", ".join([p["name"] for p in fnData["signature"]["parameters"]])}: {lambdaCode}"""
		moduleKey = "LAMBDA_FN"
	else:
		shellStr = f"""def {fnData["name"]}({", ".join([p["name"] for p in fnData["signature"]["parameters"]])}):\n{fnData["code"]}"""

		moduleKey = fnData["name"]

	return compileFunctionFromString(
		shellStr,
		fnData["name"],
		fnData["isLambda"],
		fnGlobals)

	c = compile(shellStr, "<string>", "exec")
	moduleName = "testmod"
	newModule = ModuleType(moduleName)
	exec(c, newModule.__dict__)

	for k, v in newModule.__dict__.items():
		if k == "__builtins__":
			continue
		#print(k, v)


	return newModule.__dict__[moduleKey]



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






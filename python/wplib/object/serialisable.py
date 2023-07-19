from __future__ import annotations


import json, tempfile, traceback, pprint, builtins, ast, inspect, os
import typing as T
from typing import Any
from types import ModuleType, FunctionType
from dataclasses import is_dataclass, asdict
from pathlib import PurePath, Path

from collections import defaultdict, namedtuple
from enum import Enum


import pickle # hate to do it, but for lower-level objects it's way easier

from wplib.object.reference import TypeReference
# type references should be replaced with a better overall system
from wplib.inheritance import superClassLookup
from wplib.object.visit import Visitor, SEQ_TYPES, MAP_TYPES

from wplib.constant import LITERAL_TYPES

"""Formalising serialise / deserialise interfaces.
converts live python object to dict,
and regenerates one from dict.
should be compatible with json as well
"""


class SerialisedDataKeys:
	serialisedTypeKey = "@_type"
	serialisedCommonIdentifier = "@_serialCommon"
	serialisedInterfaceIdentifier = "@_serialWrap"
	serialisedPickleIdentifier = "@_pickle"

_SERIAL_MODULE_NAME = "serialFunctionModule"



def get_short_lambda_source(lambda_func):
	"""Return the source of a (short) lambda function.
	If it's impossible to obtain, returns None.

	taken from Xion on github, left here for reference, not used
	"""
	try:
		source_lines, _ = inspect.getsourcelines(lambda_func)
	except (IOError, TypeError):
		return None

	# skip `def`-ed functions and long lambdas
	if len(source_lines) != 1:
		return None

	source_text = os.linesep.join(source_lines).strip()

	# find the AST node of a lambda definition
	# so we can locate it in the source code
	source_ast = ast.parse(source_text)
	lambda_node = next((node for node in ast.walk(source_ast)
	                    if isinstance(node, ast.Lambda)), None)
	if lambda_node is None:  # could be a single line `def fn(x): ...`
		return None

	# HACK: Since we can (and most likely will) get source lines
	# where lambdas are just a part of bigger expressions, they will have
	# some trailing junk after their definition.
	#
	# Unfortunately, AST nodes only keep their _starting_ offsets
	# from the original source, so we have to determine the end ourselves.
	# We do that by gradually shaving extra junk from after the definition.
	lambda_text = source_text[lambda_node.col_offset:]
	lambda_body_text = source_text[lambda_node.body.col_offset:]
	min_length = len('lambda:_')  # shortest possible lambda expression
	while len(lambda_text) > min_length:
		try:
			# What's annoying is that sometimes the junk even parses,
			# but results in a *different* lambda. You'd probably have to
			# be deliberately malicious to exploit it but here's one way:
			#
			#     bloop = lambda x: False, lambda x: True
			#     get_short_lamnda_source(bloop[0])
			#
			# Ideally, we'd just keep shaving until we get the same code,
			# but that most likely won't happen because we can't replicate
			# the exact closure environment.
			code = compile(lambda_body_text, '<unused filename>', 'eval')

			# Thus the next best thing is to assume some divergence due
			# to e.g. LOAD_GLOBAL in original code being LOAD_FAST in
			# the one compiled above, or vice versa.
			# But the resulting code should at least be the same *length*
			# if otherwise the same operations are performed in it.
			if len(code.co_code) == len(lambda_func.__code__.co_code):
				return lambda_text
		except SyntaxError:
			pass
		lambda_text = lambda_text[:-1]
		lambda_body_text = lambda_body_text[:-1]

	return None

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

def _enumSerialData(obj:Enum):
	return obj.value
def _enumDeserialise(cls:T.Type[Enum], serialData):
	return cls._value2member_map_[serialData]

def _defaultDictSerialData(obj: defaultdict) -> tuple:
	return (obj.default_factory, tuple(obj.items()))
def _defaultDictDeserialise(objType, serialData:tuple)->defaultdict:
	newDict = objType(serialData[0])
	for k, v in serialData[1]:
		newDict[k] = v
	return newDict

def _pathSerialData(obj: Path) -> str:
	return str(obj)
def _pathDeserialise(objType, serialData:str)->Path:
	return objType(serialData)

def _dataclassSerialData(obj):
	return asdict(obj)
def _dataclassDeserialise(objType, serialData):
	return objType(**serialData)
# check for instance of dataclass, disallow type
_isDataclassInstanceLemma = lambda obj: is_dataclass(obj) and not isinstance(obj, type)
_isDataclassTypeLemma = lambda obj, data: is_dataclass(obj)

# general methods for turning into strings
def _toStringSerialData(obj):
	return str(obj)
def _fromStringDeserialise(objType, serialData):
	return objType(serialData)

def _toPickleSerialData(obj):
	return pickle.dumps(obj)
def _fromPickleDeserialise(objType, serialData):
	return pickle.loads(serialData)

class SerialRegister:
	"""object managing methods known for serialising
	and deserialising complex types -
	in order of preference:
	- inherit from Serialisable mixin
	- implement serialise() / deserialise() interface
	- add support for your type here

	this should really just be for builtins and stl support,
	will add more integrations here as I encounter them
	"""

	typeSerialFnMap = {
		Enum : (_enumSerialData, _enumDeserialise),
		# Exception : (_exceptionSerialData, _exceptionDeserialise)
	}

	# more general lemmas called if specific type is not found
	# need separate filters for serialisation and deserialisation
	typeLemmaSerialiseFnMap = {
		_isDataclassInstanceLemma : _dataclassSerialData,
	}

	# deserialise lemmas functions are passed type and serial data
	typeLemmaDeSerialiseFnMap = {
		_isDataclassTypeLemma : _dataclassDeserialise,
	}

	@classmethod
	def registerSerialiseFnsForType(cls, type, serialDataFn, deserialiseFn):
		cls.typeSerialFnMap[type] = (serialDataFn, deserialiseFn)

	@classmethod
	def registerSerialiseFnsForLemma(cls, serialLemmaFn, serialDataFn,
	                                 deserialiseLemmaFn, deserialiseFn):
		cls.typeLemmaSerialiseFnMap[serialLemmaFn] = serialDataFn
		cls.typeLemmaDeSerialiseFnMap[deserialiseLemmaFn] = deserialiseFn


SerialRegister.registerSerialiseFnsForType(defaultdict,
                            _defaultDictSerialData,
                            _defaultDictDeserialise)
SerialRegister.registerSerialiseFnsForType(PurePath,
                            _pathSerialData,
                            _pathDeserialise)

toStringTypes = (BaseException, traceback.TracebackException, traceback.StackSummary, traceback.FrameSummary)


allPrimTypes = (*MAP_TYPES, *SEQ_TYPES)

# testing duck typing for "serialise", "deserialise" object methods instead of instance checking
# like a real python

def isSerialisable(obj):
	try: # check for attributes
		obj.serialise
		type(obj).deserialise
		return True
	except:
		return False

def isDeserialisable(data):
	try: # check for specific string tag first in data
		return data[0] == SerialisedDataKeys.serialisedInterfaceIdentifier
	except:
		return False

def serialiseObject(obj)->tuple:
	"""general function - """

	# first check for type
	if isinstance(obj, type):
		typeRef = TypeReference(obj)
		return (
			SerialisedDataKeys.serialisedTypeKey,
			typeRef.serialise()
		)

	typeRef = TypeReference(type(obj))
	# check if object implements serialisable interface
	if isSerialisable(obj):
		serialData = obj.serialise()
		return (
			SerialisedDataKeys.serialisedInterfaceIdentifier,
			typeRef.serialise(),
			serialData
		)

	foundFunctions = superClassLookup(
		SerialRegister.typeSerialFnMap, type(obj))
	if foundFunctions:
		serialData = foundFunctions[0](obj)
		return (
			SerialisedDataKeys.serialisedCommonIdentifier,
			typeRef.serialise(),
			serialData
		)

	# check through lemma functions
	for lemma, fn in SerialRegister.typeLemmaSerialiseFnMap.items():
		if lemma(obj):
			serialData = fn(obj)
			return (
				SerialisedDataKeys.serialisedCommonIdentifier,
				typeRef.serialise(),
				serialData
			)

	# fall back to pickle
	return (
		SerialisedDataKeys.serialisedPickleIdentifier,
		typeRef.serialise(),
		_toPickleSerialData(obj)
	)

	raise LookupError(f"No serialisation functions defined for type {type(obj)}")

def visitSerialiseObject(visitor, obj, parentObj):
	# check for exact typing for primitive structures
	if type(obj) in allPrimTypes:
		return obj
	result = serialiseObject(obj)
	return visitor.visitRecursive(
		result,
		parentObj
	)

def deserialiseObject(data):
	serialTag = data[0]
	loadedType = TypeReference.deserialise(data[1]).resolve()
	if serialTag == SerialisedDataKeys.serialisedTypeKey:
		return loadedType
	elif serialTag == SerialisedDataKeys.serialisedInterfaceIdentifier:
		return loadedType.deserialise(data[2])
	elif serialTag == SerialisedDataKeys.serialisedPickleIdentifier:
		# fall back to pickle
		return _fromPickleDeserialise(loadedType, data[2])

	foundFunctions = superClassLookup(
		SerialRegister.typeSerialFnMap, loadedType)
	if foundFunctions:
		return foundFunctions[1](loadedType, data[2])

	# check through lemmas
	for lemma, deserialiseFn in SerialRegister.typeLemmaDeSerialiseFnMap.items():
		if lemma(loadedType, data):
			return deserialiseFn(loadedType, data[2])


	raise LookupError(f"No deserialisation functions defined for type {type(loadedType)}")




def isSerialisedData(data):
	"""check if given data is tuple, representing serialised data"""
	try:
		return data[0] in (
			SerialisedDataKeys.serialisedTypeKey,
			SerialisedDataKeys.serialisedCommonIdentifier,
			SerialisedDataKeys.serialisedInterfaceIdentifier
		)
	except (TypeError, IndexError, KeyError):
		return False

def visitDeserialiseObject(visitor, obj, parentObj):
	#print("ds", obj, isSerialisedData(obj))
	if isSerialisedData(obj):
		return deserialiseObject(obj)
	return obj

# check for instance of dataclass, disallow type
_isDataclassInstanceLemma = lambda obj: is_dataclass(obj) and not isinstance(obj, type)

class Serialisable:
	"""
	Mixin for objects that must be transcoded to and from simple dictionaries,
	for data storage and retrieval

	Retrieving types relies on the TypeReference system
	"""

	# region override functions
	def serialise(self)->dict:
		"""override and update the result of superclasses serialise()
		result should be flattened to dict - a recursive visitor will
		run over the result to flatten it further
		dict returned by this method will be passed to deserialise()
		on loading.

		don't return

		"""
		raise NotImplementedError

	@classmethod
	def deserialise(cls, data:dict)->cls:
		"""given datamap, return instance of cls
		reinitialised with data

		"""
		raise NotImplementedError

	# general entrypoint method to avoid importing visitor classes
	def serialiseFull(self)->dict:
		"""main method - do not override"""
		return serialise(self)

	@classmethod
	def deserialiseFull(cls, data)->Serialisable:
		"""main method - do not override"""
		return deserialise(data)

def serialise(obj)->(dict, list):
	"""general entrypoint to reduce a complex structure of types to primitive maps and sequences"""
	return SerialiseVisitor.serialise(obj)

def deserialise(data:(dict, list, str))->object:
	"""general entrypoint to regenerate a complex structure
	from flattened data"""
	print("begin deserialise", type(data))
	if isinstance(data, str):
		data = ast.literal_eval(data)
	return SerialiseVisitor.deserialise(data)


# lemma to skip visiting literal types - just saves time
literalSkipLemma = lambda x: isinstance(x, LITERAL_TYPES) or x is None

class SerialiseVisitor(Visitor):
	"""holder object for the two serialisation functions"""
	@classmethod
	def serialise(cls, obj:Serialisable)->dict:
		"""creates a new visitor object, executes it on the given object"""
		visitor = cls(
			visitFn=visitSerialiseObject,
			skipLemmaSet={literalSkipLemma}
					)
		return visitor.visitRecursive(obj)
	@classmethod
	def deserialise(cls, data:dict)->(Serialisable, dict, list):
		"""creates a new visitor object, executes it on the given object"""
		visitor = cls(
			visitFn=visitDeserialiseObject,
			skipLemmaSet={literalSkipLemma}
		)
		return visitor.visitRecursive(data)


class SerialisableJsonEncoder(json.JSONEncoder):

	def default(self, o: Any) -> Any:
		if isinstance(o, Serialisable):
			return SerialiseVisitor.serialise(o)

		return super().default(o)


class SerialisableJsonDecoder(json.JSONDecoder):

	def __init__(self, localTypes=(), *args, **kwargs):
		super().__init__(*args, object_hook=self.objectHook, **kwargs)
		self.localTypes = localTypes
		self.localTypeMap = {i.__name__: i for i in localTypes}


	def objectHook(self, obj:dict):
		return SerialiseVisitor.deserialise(obj)


if __name__ == '__main__':
	print(getattr(Path(__file__), "__fspath__"))
	baseObj = {"a" : ["list"],
	           "b" : ("tuple", ),
	           "c" : Path(__file__)}
	sata = SerialiseVisitor.serialise(baseObj)
	pprint.pprint(sata)
	print(json.dumps(sata))

	regen = SerialiseVisitor.deserialise(sata)
	pprint.pprint(regen)


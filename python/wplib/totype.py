from __future__ import annotations

import traceback
import types, typing as T
from typing import overload, TypeVar, SupportsIndex, Sequence, Iterator, Any, \
	Protocol, runtime_checkable
import pprint
from collections import defaultdict
import functools, inspect, builtins
from functools import partial, partialmethod, wraps

from wplib import log

from wplib.object import UserDecorator
from wplib import log
from wplib.inheritance import containsSuperClass

nx = None
try:
	import networkx as nx
except ImportError:
	print("networkx not installed, can't use chained type conversion")


def baseConvertFn(val: T.Any, toType: type, **kwargs) -> T.Any:
	try:
		return toType(val)
	except Exception as e:
		raise TypeError(f"Basic convert cannot convert {val} of type {str} to"
		                f" {toType};\n original error:\n{traceback.format_exc()}") from e



class ToType:
	edgeRegister : dict[type, dict[type, ToType]] = defaultdict(dict)
	typeGraph = nx.DiGraph()
	typeCache : dict[tuple[type, type], ToType] = {}

	# path cache uses a single compiled lambda function to give the desired type
	pathCache : dict[tuple[type, type], T.Callable[[T.Any], T.Any]] = {}

	def __init__(self,
				 fromTypes: tuple[type, ...],
				 toTypes: tuple[type, ...],
				 convertFn: T.Callable[[T.Any, T.Type, T.ParamSpecKwargs], T.Any] = baseConvertFn,
				 backFn:T.Optional[T.Callable[[T.Any, T.Type, T.ParamSpecKwargs], T.Any]]=None
				 ):
		self.fromTypes = fromTypes if isinstance(fromTypes, T.Sequence) else (fromTypes, )
		self.toTypes = toTypes if isinstance(toTypes, T.Sequence) else (toTypes, )
		self.convertFn = convertFn
		self.updateRegisters()

		# for conversions that reverse, no point redeclaring another instance
		if callable(backFn): # create a separate sub-object with params flipped
			ToType(fromTypes=self.toTypes,
				   toTypes=self.fromTypes,
				   convertFn=backFn,
				   backFn=None)
		elif backFn:
			ToType(fromTypes=self.toTypes,
			       toTypes=self.fromTypes,
			       convertFn=self.convertFn)

	def updateRegisters(self):
		"""update this exact edge in type map"""
		for i in self.fromTypes:
			for n in self.toTypes:
				self.edgeRegister[i][n] = self
				self.typeGraph.add_edge(i, n, toType=self)

	@classmethod
	def getMatchingConvertFn(cls, fromType:type, toType:type)->T.Callable[[T.Any, type, T.ParamSpecKwargs], T.Any]:
		"""I had to do it to em
		if a direct edge is not found, try and look up through any
		matching types to see if there's a conversion path
		we can take

		a path is returned as ( (ToType, type, kwargDict), (ToType, type, etc) ... )
		"""
		if test := cls.typeCache.get((fromType, toType)):
			return test.convertFn
		if test := cls.pathCache.get((fromType, toType)): # return compiled lambda from path cache
			return test

		# check that start and end types exist in graph
		foundSrcSuperType = containsSuperClass(tuple(cls.typeGraph), fromType)
		assert foundSrcSuperType, f"no src type {fromType} to {toType}"
		foundDstSuperType = containsSuperClass(tuple(cls.typeGraph), toType)
		assert foundDstSuperType, f"no dst type {toType} from {fromType}"

		path = nx.shortest_path(
			cls.typeGraph,
			source=foundSrcSuperType,
			target=foundDstSuperType
		)
		if len(path) == 2: # direct edge found
			edgeObj = cls.typeGraph.edges[fromType, toType]["toType"]
			cls.typeCache[(fromType, toType)] = edgeObj
			return edgeObj.convertFn

		# build list of convert functions and stepping-stone types to
		# convert through
		log("graph nodes", list(cls.typeGraph))
		log("found path", path)
		stepList = []
		for i in range(1, len(path)):
			edgeObj = cls.typeGraph.edges[path[i-1], path[i]]["toType"]
			stepList.append((path[i], edgeObj))

		def _convertFn(v, t, **kwargs):
			# pass the given target type only to the final function
			log("convertFn", v, t)
			log(stepList)
			for i in stepList[:-1]:
				v = i[1].convertFn(v, i[0], **kwargs)
			return stepList[-1][1].convertFn(v, t, **kwargs)
		cls.pathCache[(fromType, toType)] = _convertFn
		return _convertFn


def to(val, t: type, **kwargs) -> t:
	if type(val) == t:
		return val
	foundFn = ToType.getMatchingConvertFn(type(val), t)
	return foundFn(val, t, **kwargs)

ToType(str, int, backFn=True)
ToType(str, float, backFn=True)
ToType(int, float, backFn=True)


"""below copied from useful_types"""
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
# Source from https://github.com/python/typing/issues/256#issuecomment-1442633430
# This works because str.__contains__ does not accept object (either in typeshed or at runtime)
@runtime_checkable
class SeqNotStr(Protocol[_T_co]):
	@overload
	def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...
	@overload
	def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...
	def __contains__(self, value: object, /) -> bool: ...
	def __len__(self) -> int: ...
	def __iter__(self) -> Iterator[_T_co]: ...
	def index(self, value: Any, start: int = 0, stop: int = ..., /) -> int: ...
	def count(self, value: Any, /) -> int: ...
	def __reversed__(self) -> Iterator[_T_co]: ...


# testS = "ada"
# testL = ["afaa"]
# log(isinstance(testS, SeqNotStr))
# log(isinstance(testL, SeqNotStr))

toSeqEdge = ToType(
	(T.Any, ),
	(list, ),
	lambda v, t, **kwargs: [v],
	backFn=lambda v, t, **kwargs:v[0] if len(v) == 1 else TypeError(
		"Cannot convert multiple-item sequence to single item"
	)
)
toTupleEdge = ToType(
	(T.Any, ),
	(tuple, ),
	lambda v, t, **kwargs: (v, ),
	backFn=lambda v, t, **kwargs:v[0] if len(v) == 1 else TypeError(
		"Cannot convert multiple-item tuple to single item"
	)
)

def _coerceToGenericList(
		val:T.Any, t:type,
		letTupleList=True,
		tupleEllipsisExtends=True,
		**convertKwargs
):
	if isinstance(val, list):
		pass
	else:
		val = [val]
	if getattr(t, "__args__", None) is None:
		return val
	# list only does single arg (but might be a union)
	argT = t.__args__[0]
	for i, el in enumerate(val):
		val[i] = coerceToGeneric(el, argT)
	return val

def _coerceToGenericTuple(
		val:T.Any, t:type,
		tupleEllipsisExtends=True,
		**convertKwargs)->tuple:
	if isinstance(val, tuple):
		pass
	else:
		val = (val,)
	if getattr(t, "__args__", None) is None:
		return val
	# if only single type specified for tuple, allow extending it
	tArgs = t.__args__
	if len(tArgs) == 1:
		return tuple(coerceToGeneric(v, tArgs[0]) for v in val)
	# check for ellipsis in tuple to extend previous arg
	newVs = [None] * len(val)
	extendIndex = -1
	argTsToCheckFromEnd = 0  # go from end after ellipsis
	for i, tArg in enumerate(tArgs):
		if tArg == ...:
			extendIndex = i
			argTsToCheckFromEnd = len(tArgs) - i
			break
	if not extendIndex and (len(tArgs) != len(val)):
		raise TypeError(
			f"Cannot coerce tuple {val} to type {t}; expected {len(tArgs)} args")

	startChunk = extendIndex
	if extendIndex == -1:
		startChunk = len(val)
	for i in range(startChunk):
		newVs[i] = coerceToGeneric(val[i], tArgs[i])
	for midI in range(extendIndex, len(val) - argTsToCheckFromEnd):
		if extendIndex <= 0 or not tupleEllipsisExtends:
			newVs[midI] = val[midI]
		else:  # coerce to previous
			newVs[midI] = coerceToGeneric(val[midI], tArgs[extendIndex - 1])

	for n in range(argTsToCheckFromEnd):
		newVs[-n - 1] = coerceToGeneric(val[-n - 1], tArgs[-n - 1])
	return tuple(newVs)


def coerceToGeneric(val:T.Any, t:type,
                    letTupleList=True,
                    tupleEllipsisExtends=True,
                    **convertKwargs
                    )->T.Any:
	"""recursively look through generic t to see if val matches -
	if not, re-run
	apart from leaf values, this is special-cased for lists and tuples

	"""
	if isinstance(t, types.GenericAlias):
		# t does not equal list if generic
		bases = types.resolve_bases(t)
		for b in bases:
			if issubclass(b, list):
				return _coerceToGenericList(val, t, letTupleList, tupleEllipsisExtends, **convertKwargs)

			if issubclass(b, tuple):
				return _coerceToGenericTuple(val, t, **convertKwargs)
	#todo: MAYBE extend to dicts here
	return to(val, t,
	          letTupleList=letTupleList,
	          tupleEllipsisExtends=tupleEllipsisExtends,
	          **convertKwargs)

class coerce(UserDecorator):
	"""Where annotations are given for function's arguments,
	coerce incoming objects to them.
	We limit recursing in objects to list and tuple

	@coerce
	def fn(a:int, b:str, c, d:(float, str):
		...

	a and b will be coerced;
	c has no given annotation;
	d has multiple, so skip it (for now)

	"""

	if T.TYPE_CHECKING:
		def __init__(self, typeConvertFnMap:dict[str, callable]=None):
			"""typeConvertFnMap is a dict of type names to functions
			which can convert to that type - default to normal To system
			"""

	@classmethod
	def getAvailableTypeNames(cls)->dict[str, type]:
		"""return dict of all available type names"""
		return {k:v for k, v in globals().items() if isinstance(v, type)}

	#
	def wrapFunction(self,
					 targetFunction:callable,
					 decoratorArgsKwargs:(None, tuple[tuple, dict])=None) ->function:
		"""convert arguments to types specified in annotations
		"""
		if T.TYPE_CHECKING:
			return targetFunction
		#log("wrapFunction", targetFunction, decoratorArgsKwargs)

		# get function annotations
		anns = targetFunction.__annotations__
		#log("anns", anns)

		# get globals of calling module
		outerFrame = inspect.currentframe().f_back
		outerGlobals = dict(outerFrame.f_globals)
		outerGlobals.update(builtins.__dict__)
		outerGlobals.update(targetFunction.__globals__)

		# dict of argument name to target type
		argNameTypeMap : dict[str, type] = {}

		for argName, argTypeStr in anns.items():
			try:
				# argtypestr is just a string, so we can't catch tuples
				# has to be eval here to catch tuples of types
				argType = eval(argTypeStr, outerGlobals)
			except (NameError, ):
				raise TypeError(
					f"Unable to find type name '{argTypeStr}' in calling module's globals:\n{outerGlobals};\nCannot coerce arguments {argName} of function {targetFunction.__name__}"
				)
			# check for tuples, skip if found
			if not isinstance(argType, types.GenericAlias):
				#log("skipping", argType, type(argType))
				continue
			argNameTypeMap[argName] = argType


		@functools.wraps(targetFunction)
		def wrapper(*args, **kwargs):
			bound = inspect.signature(targetFunction).bind(*args, **kwargs)
			nonlocal decoratorArgsKwargs
			for k, v in tuple(bound.arguments.items()):

				# if no valid annotation, skip
				if k not in argNameTypeMap:
					continue
				decoratorArgsKwargs = decoratorArgsKwargs or (None, {})
				bound.arguments[k] = coerceToGeneric(v, argNameTypeMap[k],
				                **decoratorArgsKwargs[1].get(
					                "convertKwargs", {}))
			return targetFunction(**bound.arguments)
		return wrapper



if __name__ == '__main__':

	print("begin testing")
	@coerce
	def printArgTypes(
			a:int, b:int, c, d:(float, str), e, f:str="default"
	):
		for i in locals().items():
			print(i, type(i[1]))
	printArgTypes(1, "2", 3, 4.0, 5.0, "6")

	t = list[int]
	"""generic types have:
	dict_keys(['__new__', '__repr__', '__hash__', '__call__', '__getattribute__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__iter__', '__or__', '__ror__', '__getitem__', '__mro_entries__', '__instancecheck__', '__subclasscheck__', '__reduce__', '__dir__', '__origin__', '__args__', '__unpacked__', '__parameters__', '__typing_unpacked_tuple_args__', '__doc__'])
	"""
	print(types.resolve_bases(t))
	# print(t.__dict__.keys())
	# print(type(t).__dict__.keys())
	# print(type(t).__bases__, t.__mro__)
	# print(issubclass(t, list))
	#

	# t = list[tuple[int, str]]
	# print(types.resolve_bases(t))
	#
	# print("t", t, type(t))
	# print(t.__dict__.keys())
	# print(t.__args__, t.__args__[0].__args__)

	# following raises coercion error
	#printArgTypes(1, ["2"], 3, 4.0, 5.0, "6")
	log("test deep coercion")
	@coerce
	def printArgTypes(
			a:int, b:list[tuple[int]],
	):
		for i in locals().items():
			print(i, type(i[1]))
	printArgTypes(1, "6")

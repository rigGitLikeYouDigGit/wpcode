


from __future__ import annotations

import types, pprint
import typing as T
import inspect
from types import FunctionType
from collections import defaultdict
from dataclasses import dataclass, is_dataclass, asdict, fields
from collections import deque, namedtuple
from typing import TypedDict

from wplib import log, fnlib as wfunction
from wplib.log import getDefinitionStrLink, getLineLink
#from wplib.sentinel import Sentinel
from wplib.object import Adaptor
from wplib.inheritance import superClassLookup, SuperClassLookupMap, isNamedTupleInstance, isNamedTupleClass
from wplib.object.namespace import TypeNamespace
from wplib.constant import MAP_TYPES, SEQ_TYPES, LITERAL_TYPES

if T.TYPE_CHECKING:
	from .main import VisitObjectData, DeepVisitor, visitFnType

@dataclass
class VisitPassParams:
	"""Parametres governing single iteration of visit"""
	topDown:bool = True
	depthFirst:bool = True

	visitKwargs: dict = None  # if given, kwargs to pass to visit function
	visitFn:visitFnType = None # if given, will yield (original, result) pairs or use result as new transformed object
	transformVisitedObjects: bool = False  # if true, modifies visited objects - yields (original, transformed) pairs

	passChildDataObjects:bool = False # if true, yield full ChildData objects and pass them to visitFn, not just obj
	visitRoot:bool = True # if true, visit root object
	rootObj: T.Any = None # passed as the original top object

ChildData = namedtuple("ChildData",
                       ["key", "obj", "data"],
                       defaults=[None, None, None])
#PARAMS_T = T.Dict[str, T.Any]
PARAMS_T = VisitPassParams


# test new adaptor system
class VisitAdaptor(Adaptor):
	"""adaptor for visit system - defines how to traverse and regenerate
	registered objects.

	It would make sense to combine this with pathing
	 functions for the Traversable system

	this object does not handle serialisation or text representation -
	just gives consistent interface for those systems to work
	DIFFERENT PURPOSE - move the path stuff to a later object.

	VISITOR visits objects ONCE, in CONSISTENT way - not necessarily script-friendly

	"data" is here used as a shorthand for "just do whatever, and hopefully
	this is forwards compatible enough that I don't have to rewrite it
	a 4th time "

	  """

	# abstract the structure of results away from function
	# should be able to use either format in the same visit functions

	ChildData = ChildData



	CHILD_T = ChildData
	PARAMS_T = PARAMS_T
	CHILD_LIST_T = T.Iterable[CHILD_T]
	# new base class, declare new map
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	# declare abstract methods
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		"""return iterable of (childObject, childType) pairs"""
		raise NotImplementedError("childObjects not implemented "
		                          "for type adaptor: ", cls)
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		"""create new object from base object and child type item map,
		"""
		raise NotImplementedError("newObj not implemented "
		                          "for type", type(baseObj),
		                          " adaptor: ", cls)

CHILD_T = VisitAdaptor.CHILD_T
CHILD_LIST_T = VisitAdaptor.CHILD_LIST_T
class NoneVisitAdaptor(VisitAdaptor):
	forTypes = (type(None),)
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		return ()
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		return None

class LiteralVisitAdaptor(VisitAdaptor):
	forTypes = LITERAL_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		return ()
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		return baseObj

class StrVisitAdaptor(VisitAdaptor):
	"""base to give hooks to traverse strings - different tokens,
	lines, paragraphs etc"""
	forTypes = (str,)
	def splitStrToChunks(self, s:str)->T.Iterable[str]:
		"""split string into chunks - OVERRIDE"""
		return s.split()
	def combineChunksToStr(self, chunks:T.Iterable[str])->str:
		"""combine chunks into string - OVERRIDE"""
		return "".join(chunks)
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		return ()
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		return baseObj

class MapVisitAdaptor(VisitAdaptor):
	"""we DO need a special type for dict items, since
	otherwise we lose option to capture relationship between
	keys and values -
	since we serialise separately, we don't care about
	excessive layers

	USE_TIES - should maps be considered a list of (key, value) ties,
	or literally as their basic relation?

	second is more literal and readable, but makes it more confusing to access
	dict keys specifically
	"""
	forTypes = MAP_TYPES

	@classmethod
	def templateParams(cls)->dict:
		return {"UseTies": True}

	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		"""supply an override of { dict : { "UseTies": False } } to disable ties"""
		# don't pass index, visit function can handle that
		return [ChildData(i, tie, {"childType" : "MapTie"})
		        for i, tie in enumerate(obj.items())]

	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		"""expects list of [
			( index, (key , value ), ChildType.MapItem)
			"""
		return type(baseObj)(i[1] for i in childDatas)


class SeqVisitAdaptor(VisitAdaptor):
	forTypes = SEQ_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		return (ChildData(i, val, None) for i, val in enumerate(obj))
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		#log( "  newObj", baseObj, childDatas, frames=1)
		#log("  ", type(baseObj), type(childDatas[0][1]))
		if isNamedTupleInstance(baseObj):
			return type(baseObj)(*(i[1] for i in childDatas))
		return type(baseObj)(i[1] for i in childDatas)



class Visitable:
	"""custom base interface for custom types -
	we associate an adaptor type for these later"""
	def childObjects(self, params:PARAMS_T)->CHILD_LIST_T:
		raise NotImplementedError("childObjects not implemented "
		                          "for type adaptor: ", self)
	@classmethod
	def newObj(cls, baseObj: Visitable, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		raise NotImplementedError("newObj not implemented "
		                          "for type", type(baseObj),
		                          " adaptor: ", cls)
class VisitableVisitAdaptor(VisitAdaptor):
	"""integrate derived subclasses with adaptor system
	maybe oop really was a mistake
	TODO: we can probably replace this with simple inheritance,
	 there's no need for the extra complexity"""

	forTypes = (Visitable,)
	@classmethod
	def childObjects(cls, obj:Visitable, params:PARAMS_T) ->CHILD_LIST_T:
		return obj.childObjects(params)
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		return baseObj.newObj(baseObj, childDatas, params)

class DataclassVisitAdaptor(VisitAdaptor):
	forTypes = (is_dataclass, )
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		return [ChildData(k, v, {}) for k, v in asdict(obj).items()]

	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		return type(baseObj)(**{i[0] : i[1] for i in childDatas})

from pathlib import PurePath, WindowsPath, Path, PurePosixPath, PosixPath, PureWindowsPath
class PathVisitAdaptor(VisitAdaptor):
	forTypes = (PurePath,)# WindowsPath, Path, PurePosixPath, PosixPath, PureWindowsPath)
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		return [ChildData("s", str(obj), {})]
	@classmethod
	def newObj(cls, baseObj: T.Any, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		return type(baseObj)(childDatas[0][1])
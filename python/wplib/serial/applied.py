

from __future__ import annotations
import typing as T

from enum import Enum

from wplib import CodeRef
from wplib.constant import LITERAL_TYPES, IMMUTABLE_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.serial.adaptor import SerialAdaptor
from wplib.object.namespace import TypeNamespace

"""
implementations of serialisation for standard types
"""

class LiteralSerialAdaptor(SerialAdaptor):

	uniqueAdapterName = "literal"
	forTypes = LITERAL_TYPES + (tuple, list, dict, set)

	# @classmethod
	# def encode(cls, obj, encodeParams:dict=None) ->dict:
	# 	return {"data" : obj}
	# @classmethod
	# def decode(cls, serialData:dict, decodeParams:dict=None) ->T.Any:
	# 	return serialData["data"]
	@classmethod
	def encode(cls, obj, encodeParams:dict=None) ->dict:
		return obj
	@classmethod
	def decode(cls,
	           serialData:dict,
	           serialType: type,
	           decodeParams:dict=None) ->T.Any:
		return serialData

class NamedTupleSerialAdaptor(SerialAdaptor):
	pass

class TypeAdaptor(SerialAdaptor):

	uniqueAdapterName = "type"
	forTypes = (type,)

	@classmethod
	def encode(cls, obj, encodeParams:dict=None) ->dict:
		return {"data" : CodeRef.get(obj)}
	@classmethod
	def decode(cls,
	           serialData:dict,
	           serialType:type,
	           decodeParams:dict=None) ->T.Any:
		return CodeRef.resolve(serialData["data"])


"""
now common wp types
"""


class EnumSerialAdaptor(SerialAdaptor):

	uniqueAdapterName = "stdEnum"
	forTypes = (Enum,)

	@classmethod
	def encode(cls, obj: Enum,  encodeParams:dict):
		return {"value" : obj.value, "type" : obj.__class__}

	@classmethod
	def decode(cls,
	           serialData:dict,
	           serialType:type,
	           decodeParams:dict)->Enum:
		return serialData["type"]._value2member_map_[serialData["value"]]

class TypeNamespaceAdaptor(SerialAdaptor):

	uniqueAdapterName = "typeNamespace"

	@classmethod
	def serialType(cls) ->type:
		return TypeNamespace

	@classmethod
	def encode(cls, obj, encodeParams:dict=None) ->dict:
		codeStr = CodeRef.get(obj)
		return "T:"+codeStr

	@classmethod
	def decode(cls,
	           serialData:str,
	           serialType:type,
	           decodeParams:dict=None) ->TypeNamespace:
		return CodeRef.resolve(serialData[2:])

#TODO: ###
# we REALLY should unify this with VisitAdaptor
from pathlib import PurePath
class PathSerialAdaptor(SerialAdaptor):
	forTypes = (PurePath,)# WindowsPath, Path, PurePosixPath, PosixPath, PureWindowsPath)

	@classmethod
	def encode(cls, obj, encodeParams:dict=None) ->dict:
		return str(obj)
	@classmethod
	def decode(cls,
	           serialData:dict,
	           serialType:type,
	           decodeParams:dict=None) ->T.Any:
		return serialType(serialData)

from dataclasses import dataclass, is_dataclass, asdict
class DataclassSerialAdaptor(SerialAdaptor):
	forTypes = (is_dataclass, )
	@classmethod
	def encode(cls, obj, encodeParams:dict=None) ->dict:
		return asdict(obj)
	@classmethod
	def decode(cls,
	           serialData:dict,
	           serialType:type,
	           decodeParams:dict=None) ->T.Any:
		return serialType(**serialData)



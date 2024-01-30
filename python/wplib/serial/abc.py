
from __future__ import annotations
import typing as T

from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import CodeRef, inheritance, log, dictlib
from wplib.inheritance import SuperClassLookupMap

from wplib.object.visitor import VisitObjectData, DeepVisitor
from wplib.object import VisitPassParams
from wplib.serial.ops import SerialiseOp, DeserialiseOp
from wplib.serial.adaptor import SerialAdaptor
from wplib.serial.main import serialise, deserialise

class Serialisable:
	"""Base class for serialisable objects -
	delegates encode() to instance method
	"""


	@classmethod
	def makeFormatData(cls, forObj, serialData:dict)->dict:
		"""create base format dict for given object
		"""
		return {SerialAdaptor.TYPE_KEY : CodeRef.get(forObj)}

	def defaultEncodeParams(self)->dict:
		"""Get the default encode params. These are combined with any given by user and passed to encode().
		"""
		return {}

	@classmethod
	def defaultDecodeParams(cls)->dict:
		"""Get the default decode params.
		"""
		return {}


	def encode(self, encodeParams:dict=None)->dict:
		"""Encode the given object into a dict -
		DO NOT recurse into sub-objects, visitor will handle that.
		"""
		return {}
	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None)->T.Any:
		"""Decode the object from a dict.
		"""
		raise NotImplementedError()

	# attach methods for convenience
	def serialise(self, serialiseOp=None, serialParams=None)->dict:
		"""serialise the object into a dict
		"""
		return serialise(self, serialiseOp=serialiseOp, serialParams=serialParams)
	@classmethod
	def deserialise(cls, serialData:dict, deserialiseOp=None, serialParams=None)->T.Any:
		"""deserialise the object from a dict
		"""
		return deserialise(serialData, deserialiseOp=deserialiseOp, serialParams=serialParams)


class SerialisableAdaptor(SerialAdaptor):
	"""Adaptor for serialisable objects -
	encodes and decodes using the object's encode() and decode() methods
	"""
	uniqueAdapterName = "serialisable"
	forTypes = (Serialisable,)

	@classmethod
	def defaultEncodeParams(cls, forObj) ->dict:
		return forObj.defaultEncodeParams()

	@classmethod
	def defaultDecodeParams(cls, forSerialisedType:Serialisable) ->dict:
		return forSerialisedType.defaultDecodeParams()

	@classmethod
	def encode(cls, obj:Serialisable, encodeParams:dict=None)->dict:
		"""Encode the given object into a dict.
		"""
		return {cls.FORMAT_KEY : cls.makeFormatData(obj, encodeParams),
		        cls.DATA_KEY : obj.encode(encodeParams)}

	@classmethod
	def decode(cls,
	           serialData:dict,
	           serialType: type[Serialisable],
	           decodeParams:dict=None)->T.Any:
		"""
		Decode the object from a dict.
		find correct custom type, then call its own decode method
		"""
		# foundType = CodeRef.resolve(serialData[cls.FORMAT_KEY][cls.TYPE_KEY])
		return serialType.decode(serialData[cls.DATA_KEY], decodeParams)




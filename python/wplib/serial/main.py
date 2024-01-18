from __future__ import annotations
import typing as T
import pprint

from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import CodeRef, inheritance, log
from wplib.inheritance import SuperClassLookupMap

from .adaptor import SerialAdaptor


from wplib.object.visitor import VisitObjectData, DeepVisitor
from ..object import VisitPassParams

#from .encoder import EncoderBase
#from . import lib


"""global register for adaptor classes - top-level
interface for serialisation system.

Adaptors must be registered explicitly - they also must not
collide, only one adaptor per class.
"""


# serialise from root down
# deserialise from leaves up

class SerialiseOp(DeepVisitor.DeepVisitOp):

	def visit(self,
	          obj:T.Any,
	              visitor:DeepVisitor=None,
	              visitData:VisitObjectData=None,
	              visitPassParams:VisitPassParams=None,
	              ) ->T.Any:
		"""Transform to apply to each object during serialisation.
		affects only single layer of object, visitor handles continuation
		"""
		serialParams: dict = visitPassParams.visitKwargs["serialParams"]
		if obj is None:
			return None
		# get adaptor for the given data
		adaptorCls = SerialAdaptor.adaptorForType(type(obj))
		if adaptorCls is None:
			raise Exception(f"No adaptor for class {type(obj)}")
		#log("adaptor", adaptorCls)
		return adaptorCls.encode(obj, encodeParams=serialParams)

class DeserialiseOp(DeepVisitor.DeepVisitOp):
	@classmethod
	def visit(cls,
	          obj:T.Any,
	              visitor:DeepVisitor=None,
	              visitObjectData:VisitObjectData=None,
	              visitPassParams:VisitPassParams=None,
	              ) ->T.Any:

		"""Transform to apply to each object during deserialisation.
		"""
		print("deserialise", obj)

		serialParams : dict = visitPassParams.visitKwargs["serialParams"]

		if obj is None:
			return None

		if not isinstance(obj, dict):
			return obj

		if obj.get(SerialAdaptor.FORMAT_KEY, None) is None:
			return obj

		# reload serialised type and get the adaptor for it
		adaptorCls = SerialAdaptor.adaptorForData(obj[SerialAdaptor.FORMAT_KEY])

		if adaptorCls is None:
			raise Exception(f"No adaptor for class {type(obj)}")
		# decode the object
		log("decode", obj)
		return adaptorCls.decode(obj, decodeParams=serialParams)


def serialise(obj, serialiseOp=None, serialParams=None)->dict:
	"""top-level serialise function -
	serialises the given object into a dict
	"""
	visitor = DeepVisitor()
	serialiseOp = serialiseOp or SerialiseOp()
	serialParams = serialParams or {}
	params = VisitPassParams(
		visitFn=serialiseOp.visit,
		visitKwargs={
			"serialParams": serialParams,
		},
		transformVisitedObjects=True,
	)
	result = visitor.dispatchPass(
		obj,
		params,
	)
	return result

def deserialise(serialData:dict, deserialiseOp=None, serialParams=None)->T.Any:
	"""top-level deserialise function -
	deserialises the given dict into an object
	"""
	visitor = DeepVisitor()
	deserialiseOp = deserialiseOp or DeserialiseOp()
	serialParams = serialParams or {}
	params = VisitPassParams(
		visitFn=deserialiseOp.visit,
		visitKwargs={
			"serialParams": serialParams,
		},
		transformVisitedObjects=True,
		topDown=False # deserialise from leaves up
	)
	result = visitor.dispatchPass(
		serialData,
		params,
	)
	return result

class Serialisable:
	"""Base class for serialisable objects -
	delegates encode() to instance method
	"""


	@classmethod
	def makeFormatData(cls, forObj, serialData:dict)->dict:
		"""create base format dict for given object
		"""
		return {SerialAdaptor.TYPE_KEY : CodeRef.get(forObj)}

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
	def encode(cls, obj:Serialisable, encodeParams:dict=None)->dict:
		"""Encode the given object into a dict.
		"""
		return {cls.FORMAT_KEY : cls.makeFormatData(obj, encodeParams),
		        cls.DATA_KEY : obj.encode(encodeParams)}

	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None)->T.Any:
		"""
		Decode the object from a dict.
		find correct custom type, then call its own decode method
		"""
		foundType = CodeRef.resolve(serialData[cls.FORMAT_KEY][cls.TYPE_KEY])
		return foundType.decode(serialData[cls.DATA_KEY], decodeParams)






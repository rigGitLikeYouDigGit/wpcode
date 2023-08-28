from __future__ import annotations


from wplib.object.visit import recursiveVisitCopy, visitTopDown, visitLeavesUp
from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import CodeRef

from .adaptor import SerialAdaptor
#from .encoder import EncoderBase
from .register import SerialRegister
#from . import lib


# serialise from root down
# deserialise from leaves up

def serialiseTransform(obj, visitData=None):
	"""Transform to apply to each object during serialisation.
	"""
	if obj is None:
		return None
	# literals can be serialised just so
	if isinstance(obj, LITERAL_TYPES):
		return obj
	# containers will be visited by outer function
	if isinstance(obj, MAP_TYPES + SEQ_TYPES):
		return obj

	if isinstance(obj, SerialAdaptor):
		return obj.encode(obj)

	# retrieve adaptor for the given data
	adaptorCls = SerialRegister.adaptorForClass(type(obj))
	if adaptorCls is None:
		raise Exception(f"No adaptor for {obj}, class {type(obj)}")
	return adaptorCls.encode(obj)


def serialiseRecursive(obj, ):
	"""Serialise the given object, recursively visiting
	any objects that are Visitable.
	"""
	return visitTopDown(serialiseTransform(obj), serialiseTransform)

def deserialiseTransform(obj, visitData=None):
	"""Transform to apply to each object during deserialisation.
	"""
	#print("deserialise", obj)
	# literals can be serialised just so
	if isinstance(obj, LITERAL_TYPES):
		return obj

	if isinstance(obj, MAP_TYPES):
		#print("get", lib.getDataCodeRefStr(obj))
		codeRef = SerialAdaptor.getDataCodeRefStr(obj)
		if codeRef is not None:
			serialType = CodeRef.resolve(codeRef)
			#print(f"Found code ref {codeRef} -> {serialType}")
			if issubclass(serialType, SerialAdaptor):
				return serialType.decode(obj)
			# retrieve adaptor for the given data
			adaptorCls = SerialRegister.adaptorForData(obj)
			if adaptorCls is None:
				raise Exception(f"No adaptor for class {type(obj)}")
			return adaptorCls.decode(obj)

	return obj



def deserialiseRecursive(data:dict):
	"""Deserialise the given data, recursively visiting
	any objects that are Visitable.
	"""
	return visitLeavesUp(data, deserialiseTransform)


class Serialisable(SerialAdaptor):
	"""For custom complex types, define the relevant encoding and decoding
	alongside the main class
	"""
	# EncoderBase = EncoderBase
	# @version(1)
	# class Encoder(EncoderBase):
	# 	"""Specialise encoder versions here for this class
	# 	"""
	# 	encodeType = None
	#
	# 	@classmethod
	# 	def encode(cls, obj:Serialisable):
	# 		"""serialise a single level of outer object
	# 		called from top down"""
	# 		raise NotImplementedError
	#
	# 	@classmethod
	# 	def decode(cls, serialCls:type[Serialisable], serialData:dict) ->Serialisable:
	# 		"""deserialise a single level of outer object
	# 		called from leaves up"""
	# 		raise NotImplementedError


	def serialise(self, **kwargs)->dict:
		"""top level method to get serialised representation"""
		return serialiseRecursive(self)

	@classmethod
	def deserialise(cls, data:dict)->Serialisable:
		"""top level method to get deserialised representation"""
		return deserialiseRecursive(data)

	# def __init_subclass__(cls, **kwargs):
	# 	super(Serialisable).__init_subclass__(**kwargs)
	# 	# register the class
	# 	cls.encoderBaseCls = cls.EncoderBase
	# 	SerialRegister.registerAdaptor(cls)





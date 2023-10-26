from __future__ import annotations
import typing as T


from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import CodeRef, inheritance, log
from wplib.inheritance import SuperClassLookupMap

from .adaptor import SerialAdaptor

from wplib.object.visitor import visitFunctionRegister, VisitObjectData, DeepVisitor
from ..object import VisitPassParams

#from .encoder import EncoderBase
#from . import lib


"""global register for adaptor classes - top-level
interface for serialisation system.

Adaptors must be registered explicitly - they also must not
collide, only one adaptor per class.
"""

class SerialRegister:
	"""going with one single global class here -
	if you need crazy behaviour overrides, subclass this Register
	and re-register all the builtin types yourself
	"""

	# map of { serial type : adaptor for that type }
	typeAdaptorMap = SuperClassLookupMap({})

	@classmethod
	def registerAdaptor(cls, adaptor:T.Type[SerialAdaptor]):
		adaptor.checkIsValid(), f"Adaptor {adaptor} is not valid"

		items = set(cls.typeAdaptorMap.classMap.items())

		if (adaptor.serialType(), adaptor) in items: # exact pair already registered
			return

		assert adaptor.serialType() not in cls.typeAdaptorMap.classMap, f"Adaptor {adaptor} already registered"

		cls.typeAdaptorMap.updateClassMap({adaptor.serialType() : adaptor})

	@classmethod
	def adaptorForClass(cls, forCls:T.Type)->SerialAdaptor:
		"""Get the adaptor for the given class.
		"""
		result = cls.typeAdaptorMap.lookup(forCls, default=None)
		if result is None:
			print("missing:")
			print(cls.typeAdaptorMap)
			print(forCls)
			print(inheritance.superClassLookup(cls.typeAdaptorMap, forCls, default=None))
		return result

	# @classmethod
	# def adaptorForCodeRefString(cls, codeRefString:str)->SerialAdaptor:
	# 	"""Get the adaptor for the given code ref string.
	# 	We rely on the code ref still being valid here - improve later if needed.
	# 	Maybe there's a benefit to registering adaptors alongside their own
	# 	code ref paths, so the class has a record of previous places it was defined
	# 	"""
	# 	return cls.adaptorForClass(CodeRef.resolve(codeRefString))

	@classmethod
	def adaptorForData(cls, data:dict)->SerialAdaptor:
		"""Get the adaptor for the given data dict.
		"""
		return cls.adaptorForClass(CodeRef.resolve(SerialAdaptor.getDataCodeRefStr(data)))


# serialise from root down
# deserialise from leaves up

class SerialiseOp(DeepVisitor.DeepVisitOp):

	@classmethod
	def visit(cls,
	          obj:T.Any,
	              visitor:DeepVisitor=None,
	              visitData:VisitObjectData=None,
	              visitParams:VisitPassParams=None,
	              ) ->T.Any:
		"""Transform to apply to each object during serialisation.
		affects only single layer of object, visitor handles continuation
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


# serialiseVisitor = DeepVisitor(
# 	SerialiseOp.visit
# )

def serialiseRecursive(obj)->dict:
	"""top-level function to set off visit pass"""
	visitParams = DeepVisitor.VisitPassParams(
		topDown=True,
		depthFirst=True,
		runVisitFn=True,
		transformVisitedObjects=True,
		visitFn=SerialiseOp.visit
	)
	return DeepVisitor().dispatchPass(obj, visitParams)





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



# def deserialiseRecursive(data:dict):
# 	"""Deserialise the given data, recursively visiting
# 	any objects that are Visitable.
# 	"""
# 	return visitLeavesUp(data, deserialiseTransform)


def deserialiseRecursive(obj:dict)->T.Any:
	"""top-level function to set off visit pass"""
	visitParams = DeepVisitor.VisitPassParams(
		topDown=False,
		depthFirst=True,
		runVisitFn=True,
		transformVisitedObjects=True,
		visitFn=DeserialiseOp.visit
	)
	return DeepVisitor().dispatchPass(obj, visitParams)





class Serialisable(SerialAdaptor):
	"""For custom complex types, define the relevant encoding and decoding
	alongside the main class
	"""

	@classmethod
	def serialType(cls) ->type:
		return cls

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

	def __init_subclass__(cls, **kwargs):
		super(Serialisable).__init_subclass__(**kwargs)
		# register the class
		SerialRegister.registerAdaptor(cls)
		log("registered", cls)





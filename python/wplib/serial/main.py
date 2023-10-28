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
		#adaptor.checkIsValid(), f"Adaptor {adaptor} is not valid"

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
	              visitPassParams:VisitPassParams=None,
	              ) ->T.Any:
		"""Transform to apply to each object during serialisation.
		affects only single layer of object, visitor handles continuation
		"""
		serialParams: dict = visitPassParams.visitKwargs["serialParams"]
		#print("serialise", obj)
		if obj is None:
			return None
		# literals can be serialised just so
		if isinstance(obj, LITERAL_TYPES):
			return obj
		# containers will be visited by outer function
		if isinstance(obj, MAP_TYPES + SEQ_TYPES):
			return obj

		if isinstance(obj, SerialAdaptor):
			return obj.encode(obj, encodeParams=serialParams)

		# retrieve adaptor for the given data
		adaptorCls = SerialRegister.adaptorForClass(type(obj))
		if adaptorCls is None:
			raise Exception(f"No adaptor for {obj}, class {type(obj)}")
		return adaptorCls.encode(obj, encodeParams=serialParams)


# serialiseVisitor = DeepVisitor(
# 	SerialiseOp.visit
# )

# def serialiseRecursive(obj, params:dict)->dict:
# 	"""top-level function to set off visit pass"""
# 	visitParams = DeepVisitor.VisitPassParams(
# 		topDown=True,
# 		depthFirst=True,
# 		runVisitFn=True,
# 		transformVisitedObjects=True,
# 		visitFn=SerialiseOp.visit
# 	)
# 	#print("serialise", obj)
# 	return DeepVisitor().dispatchPass(obj, visitParams)





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

		serialParams : dict = visitPassParams.visitKwargs["serialParams"]

		# literals can be serialised just so
		if isinstance(obj, LITERAL_TYPES):
			return obj

		if isinstance(obj, MAP_TYPES):
			#print("get", CodeRef.getDataCodeRefStr(obj))
			codeRef = SerialAdaptor.getDataCodeRefStr(obj)
			#log("code ref", codeRef)
			if codeRef is not None: # found ref, deserialise type

				visitObjectData["makeNewObjFromVisitResult"] = False

				serialType = CodeRef.resolve(codeRef)
				#log(f"Found code ref {codeRef} -> {serialType}")
				if issubclass(serialType, SerialAdaptor):
					#log("is serial adaptor", serialType)
					result = serialType.decode(obj, decodeParams=serialParams)
					#log("result", result)
					return result
				# retrieve adaptor for the given data
				adaptorCls = SerialRegister.adaptorForData(obj)
				if adaptorCls is None:
					raise Exception(f"No adaptor for class {type(obj)}")
				return adaptorCls.decode(obj, decodeParams=serialParams)

		return obj



# def deserialiseRecursive(data:dict):
# 	"""Deserialise the given data, recursively visiting
# 	any objects that are Visitable.
# 	"""
# 	return visitLeavesUp(data, deserialiseTransform)


# def deserialiseRecursive(obj:dict, params:dict)->T.Any:
# 	"""top-level function to set off visit pass"""
# 	visitParams = DeepVisitor.VisitPassParams(
# 		topDown=False,
# 		depthFirst=True,
# 		runVisitFn=True,
# 		transformVisitedObjects=True,
# 		visitFn=DeserialiseOp.visit
# 	)
# 	result = DeepVisitor().dispatchPass(obj, visitParams)
# 	log("deserialiseRec", result)
# 	return result





class Serialisable(SerialAdaptor):
	"""For custom complex types, define the relevant encoding and decoding
	alongside the main class
	"""

	@classmethod
	def serialType(cls) ->type:
		return cls

	#@classmethod
	def serialise(self, params:dict=None)->dict:
		"""top level method to get serialised representation"""
		obj = self
		if not isinstance(self, Serialisable):
			# called as class method on random object, serialise anyway
			params = params or Serialisable.defaultEncodeParams()
		else:
			params = params or self.defaultEncodeParams()
		visitParams = DeepVisitor.VisitPassParams(
			topDown=True,
			depthFirst=True,
			runVisitFn=True,
			transformVisitedObjects=True,
			visitFn=SerialiseOp.visit
		)
		result = DeepVisitor().dispatchPass(obj, visitParams, serialParams=params)
		#log("serialiseRec", result)
		return result
		#return serialiseRecursive(self, params)

	@classmethod
	def deserialise(cls, data:dict, params=None)->Serialisable:
		"""top level method to get deserialised representation"""
		params = params or cls.defaultDecodeParams()

		visitParams = DeepVisitor.VisitPassParams(
			topDown=False,
			depthFirst=True,
			runVisitFn=True,
			transformVisitedObjects=True,
			visitFn=DeserialiseOp.visit
		)
		result = DeepVisitor().dispatchPass(data, visitParams, serialParams=params)
		log("deserialiseRec", result)
		return result

	def __init_subclass__(cls, **kwargs):
		super(Serialisable).__init_subclass__(**kwargs)
		# register the class
		SerialRegister.registerAdaptor(cls)
		log("registered", cls)





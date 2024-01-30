
from __future__ import annotations
import typing as T

from wplib import dictlib
from wplib.object import DeepVisitor, VisitAdaptor, VisitObjectData, VisitPassParams
from wplib.serial.adaptor import SerialAdaptor


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
		serialParams: dict = dict(visitPassParams.visitKwargs["serialParams"])
		if obj is None:
			return None
		# get adaptor for the given data
		adaptorCls = SerialAdaptor.adaptorForType(type(obj))
		if adaptorCls is None:
			raise Exception(f"No adaptor for class {type(obj)}")
		#log("adaptor", adaptorCls)
		# copy params and update with class defaults
		dictlib.defaultUpdate(serialParams, adaptorCls.defaultEncodeParams(obj)
		                      )
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
		#log("deserialise", obj)

		serialParams : dict = dict(visitPassParams.visitKwargs["serialParams"])

		if obj is None:
			return None

		if not isinstance(obj, dict):
			return obj

		if obj.get(SerialAdaptor.FORMAT_KEY, None) is None:
			return obj

		# reload serialised type and get the adaptor for it
		serialisedType = SerialAdaptor.typeForData(obj[SerialAdaptor.FORMAT_KEY])
		# adaptorCls = SerialAdaptor.adaptorForData(obj[SerialAdaptor.FORMAT_KEY])
		adaptorCls = SerialAdaptor.adaptorForType(serialisedType)

		if adaptorCls is None:
			raise Exception(f"No adaptor for class {type(obj)}")
		dictlib.defaultUpdate(serialParams, adaptorCls.defaultDecodeParams(serialisedType)
		                      )
		# decode the object
		#log("decode", obj)
		return adaptorCls.decode(
			obj,
			serialType=serialisedType,
			decodeParams=serialParams)

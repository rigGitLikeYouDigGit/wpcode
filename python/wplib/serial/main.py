from __future__ import annotations
import typing as T
import pprint

from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES, MAP_TYPES, SEQ_TYPES
from wplib import CodeRef, inheritance, log, dictlib
from wplib.inheritance import SuperClassLookupMap

from .adaptor import SerialAdaptor


from wplib.object.visitor import VisitObjectData, DeepVisitor
from wplib.object import VisitPassParams
from wplib.serial.ops import SerialiseOp, DeserialiseOp

#from .encoder import EncoderBase
#from . import lib


"""global register for adaptor classes - top-level
interface for serialisation system.

Adaptors must be registered explicitly - they also must not
collide, only one adaptor per class.
"""


# serialise from root down
# deserialise from leaves up



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
	#log("deserialise result", result)
	return result




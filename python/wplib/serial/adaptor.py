
from __future__ import annotations
import typing as T

from wplib.validation import ValidationError
from wplib import CodeRef, inheritance
from wplib.object import Adaptor

from .constant import ENCODE_DATA_KEY, FORMAT_DATA_KEY



class SerialAdaptor(Adaptor):
	"""Helper class to be used with external types,
	defining our own rules for saving and loading.

	This outer type should also define the type's serial-UID,
	and hold all versioned encoders.

	Real encoders should be defined within the scope of this class

	all serial adaptors must define encode() and decode() methods
	"""

	adaptorTypeMap = Adaptor.makeNewTypeMap()

	# static to dissuade any kind of automated generation -
	# DEFINE THIS MANUALLY AND NEVER CHANGE IT
	uniqueAdapterName : str = None

	TYPE_KEY = "@TYPE"
	FORMAT_KEY = "@F"
	DATA_KEY = "@D"

	VERSION_DATA_NAME_KEY = "name"
	VERSION_DATA_VERSION_KEY = "version"
	VERSION_DATA_TYPE_KEY = "type"

	# version this up whenever a meaningful change in format is committed
	LATEST_DATA_VERSION = 1

	@classmethod
	def makeFormatData(cls, forObj, serialData:dict)->dict:
		"""create base format dict for given object
		"""
		return {cls.TYPE_KEY : CodeRef.get(forObj)}

	@classmethod
	def encode(cls, obj, encodeParams:dict=None)->dict:
		"""Encode the given object into a dict.
		"""
		raise NotImplementedError()

	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None)->T.Any:
		"""Decode the object from a dict.
		"""
		raise NotImplementedError()

	@classmethod
	def getDataCodeRefStr(cls, data: dict) -> str:
		"""Get the code ref from the given data dict.
		"""
		if not isinstance(data, dict):
			return None
		return data.get(cls.TYPE_KEY, None)

	@classmethod
	def adaptorForData(cls, data:dict)->SerialAdaptor:
		"""Get the adaptor for the given data dict.
		"""
		return cls.adaptorForType(CodeRef.resolve(SerialAdaptor.getDataCodeRefStr(data)
		                                          ))

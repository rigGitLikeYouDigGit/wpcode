
from __future__ import annotations
import typing as T

from wplib.validation import ValidationError
from wplib import CodeRef, inheritance

from .constant import ENCODE_DATA_KEY, FORMAT_DATA_KEY



class SerialAdaptor:
	"""Helper class to be used with external types,
	defining our own rules for saving and loading.

	This outer type should also define the type's serial-UID,
	and hold all versioned encoders.

	Real encoders should be defined within the scope of this class
	"""

	# static to dissuade any kind of automated generation -
	# DEFINE THIS MANUALLY AND NEVER CHANGE IT
	uniqueAdapterName : str = None

	VERSION_DATA_NAME_KEY = "name"
	VERSION_DATA_VERSION_KEY = "version"
	VERSION_DATA_TYPE_KEY = "type"

	# version this up whenever a meaningful change in format is committed
	LATEST_DATA_VERSION = 1

	@classmethod
	def serialType(cls)->type:
		"""Return the type that this adaptor serialises -
		by defult, the adaptor class itself.
		This allows inheriting from adaptor directly -
		this shouldn't be done
		"""
		raise NotImplementedError()
		#return cls

	@classmethod
	def getFormatDataToSerialise(cls, version:int, objToSerialise)->dict:
		"""Return the marker data for a given version -
		include this adapter's unique name and the target version,
		"""
		assert cls.uniqueAdapterName
		return {
			cls.VERSION_DATA_NAME_KEY : cls.uniqueAdapterName,
			cls.VERSION_DATA_VERSION_KEY : version,
			cls.VERSION_DATA_TYPE_KEY : CodeRef.get(type(objToSerialise)),
		}

	@classmethod
	def getFormatData(cls, data: dict):
		return data.get(FORMAT_DATA_KEY, {})

	@classmethod
	def getDataCodeRefStr(cls, data: dict) -> str:
		"""Get the code ref from the given data dict.
		"""
		return cls.getFormatData(data).get(cls.VERSION_DATA_TYPE_KEY, None)


	@classmethod
	def defaultEncodeParams(cls)->dict:
		return {}

	@classmethod
	def defaultDecodeParams(cls)->dict:
		return {}

	# main methods

	@classmethod
	def _encodeObject(cls, obj, encodeParams:dict):
		"""Encode the given object into a dict.
		"""
		raise NotImplementedError()

	@classmethod
	def encode(cls, obj, encodeParams:dict=None)->dict:
		"""Encode outer object into a dict - if no version is specified,
		use the latest. No recursion needed, visitor will handle that.

		"""
		encodeParams = {**cls.defaultEncodeParams(), **(encodeParams or {})}

		#print("found encoder", encoder, "for type", type(obj))
		return {
			**cls._encodeObject(obj, encodeParams),
			FORMAT_DATA_KEY : cls.getFormatDataToSerialise(cls.LATEST_DATA_VERSION,
			                                               obj)
		}

	@classmethod
	def _decodeObject(cls, serialType:type, serialData:dict, decodeParams:dict, formatVersion=-1):
		"""Decode the given object from a dict.
		if dataVersion is not specified, defaults to latest -
		add in cases here however necessary to preserve support for older data formats

		"""
		raise NotImplementedError()

	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None)->serialType():
		"""Decode the object from a dict.
		"""
		assert FORMAT_DATA_KEY in serialData, f"Serial data missing format data key {FORMAT_DATA_KEY}"
		# get the version data
		formatData = serialData[FORMAT_DATA_KEY]
		dataVersion = formatData.get(cls.VERSION_DATA_VERSION_KEY, -1)
		decodeParams = {**cls.defaultDecodeParams(), **(decodeParams or {})}

		# resolve the type
		# catch coderef exception here
		serialType = CodeRef.resolve(formatData[cls.VERSION_DATA_TYPE_KEY])

		# decode
		return cls._decodeObject(serialType, serialData, decodeParams, dataVersion)
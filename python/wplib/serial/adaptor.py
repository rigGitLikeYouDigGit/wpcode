
from __future__ import annotations
import typing as T

from wplib.validation import ValidationError
from wplib import CodeRef, inheritance

from .constant import ENCODE_DATA_KEY, FORMAT_DATA_KEY

class EncoderBase:
	@classmethod
	def encode(cls, obj:T.Any, **kwargs)->dict:
		"""Encode the given object into a dict.
		"""
		raise NotImplementedError()
	@classmethod
	def decode(cls, serialCls:type, serialData:dict)->T.Any:
		"""Decode the given dict into an object.
		"""
		raise NotImplementedError()

	@classmethod
	def getVersion(cls)->int:
		"""Return the version of this encoder.
		"""
		return cls._versionIndex


def encoderVersion(index:int):
	"""Decorate internally-defined Encoder classes with this -
	classes must define classmethods for encode() and decode().
	Beyond that, do whatever
	"""
	assert index > 0, f"Version index must be greater than 0, not {index}"
	def _inner(cls:type[EncoderBase]):
		assert "encode" in cls.__dict__, f"Encoder {cls} does not define encode()"
		assert "decode" in cls.__dict__, f"Encoder {cls} does not define decode()"
		cls.__name__ = f"{cls.__name__}_V{index}"
		cls._versionIndex = index
		cls._isEncoder = True
		return cls
	return _inner

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

	encoderVersion = encoderVersion

	@classmethod
	def serialType(cls)->type:
		"""Return the type that this adaptor serialises -
		by defult, the adaptor class itself.
		"""
		return cls

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
	def encoderVersionMap(cls)->dict[int, T.Type[EncoderBase]]:
		"""Return a map of version-index to encoder class.
		"""
		encoders = {}
		#print("get version map for", cls)
		for k, v in inheritance.mroMergedDict(cls).items():
		#for k, v in cls.__dict__.items():
			if not isinstance(v, type):
				continue
			if getattr(v, "_isEncoder", False):
				encoders[v._versionIndex] = v
		#print("returning encoders", encoders)
		return encoders

	@classmethod
	def latestEncoderVersion(cls)->int:
		"""Return the latest encoder class version.
		"""
		return max(cls.encoderVersionMap().keys())

	@classmethod
	def getEncoder(cls, versionIndex:int=None)->T.Type[EncoderBase]:
		"""by default, retrieve the latest"""
		#print("encoder map", cls.encoderVersionMap())
		if versionIndex is None:
			return cls.encoderVersionMap()[max(cls.encoderVersionMap().keys())]
		return cls.encoderVersionMap()[versionIndex]


	@classmethod
	def checkIsValid(cls)->bool:
		"""Check that the class has been defined correctly.
		"""
		# if cls.encoderBaseCls is None:
		# 	raise ValidationError(f"Encoder base class not set for {cls}")
		if not isinstance(cls.uniqueAdapterName, str) :
			raise ValidationError(f"Unique adapter name not set for {cls}")
		# if not issubclass(cls.encoderBaseCls, EncoderBase):
		# 	raise ValidationError(f"Encoder base class {cls.encoderBaseCls} is not a subclass of {EncoderBase}")
		if not cls.encoderVersionMap():
			raise ValidationError(f"No encoders defined for {cls}, no versionMap derived")
		return True

	# main methods
	@classmethod
	def encode(cls, obj, encoderVersion:int=None, **kwargs)->dict:
		"""Encode the object into a dict - if no version is specified,
		use the latest. (Latest should probably always be used when saving).

		mixing in the key for format data sets off my spidey-sense, but the alternative
		is creating a new container level of dict for each new type in a hierarchy,
		which gets tedious to read.
		"""
		encoder = cls.getEncoder(versionIndex=encoderVersion)
		#print("found encoder", encoder, "for type", type(obj))
		return {
			**encoder.encode(obj, **kwargs),
			FORMAT_DATA_KEY : cls.getFormatDataToSerialise(encoder.getVersion(),
			                                               obj)
		}

	@classmethod
	def decode(cls, serialData:dict)->serialType():
		"""Decode the object from a dict.
		"""
		assert FORMAT_DATA_KEY in serialData, f"Serial data missing format data key {FORMAT_DATA_KEY}"
		# get the version data
		formatData = serialData[FORMAT_DATA_KEY]
		# get the encoder
		encoder = cls.encoderVersionMap()[formatData[cls.VERSION_DATA_VERSION_KEY]]

		# resolve the type
		# catch coderef exception here
		serialType = CodeRef.resolve(formatData[cls.VERSION_DATA_TYPE_KEY])

		# decode
		return encoder.decode(
			serialType, serialData
		)
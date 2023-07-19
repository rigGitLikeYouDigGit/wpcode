
from __future__ import annotations
import typing as T

from wplib.validation import ValidationError
from wplib import coderef

from .constant import ENCODE_DATA_KEY, FORMAT_DATA_KEY
from .encoder import EncoderBase

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

	encoderBaseCls:T.Type[EncoderBase] = None

	VERSION_DATA_NAME_KEY = "name"
	VERSION_DATA_VERSION_KEY = "version"
	VERSION_DATA_TYPE_KEY = "type"

	@classmethod
	def serialType(cls)->T.Type:
		"""Return the type that this adaptor is for.
		"""
		return cls.encoderBaseCls.encodeType

	@classmethod
	def getFormatData(cls, version:int, objToSerialise)->dict:
		"""Return the marker data for a given version -
		include this adapter's unique name and the target version,
		"""
		assert cls.uniqueAdapterName
		return {
			cls.VERSION_DATA_NAME_KEY : cls.uniqueAdapterName,
			cls.VERSION_DATA_VERSION_KEY : version,
			cls.VERSION_DATA_TYPE_KEY : coderef.getCodeRef(type(objToSerialise)),
		}

	@classmethod
	def encoderVersionMap(cls)->dict[int, T.Type[EncoderBase]]:
		"""Return a map of version-index to encoder class.
		"""
		encoders = {}
		for k, v in cls.__dict__.items():
			if not isinstance(v, type):
				continue
			if k == "encoderBaseCls": #reference to base, not an actual encoder
				continue
			if issubclass(v, cls.encoderBaseCls):
				assert v.checkIsValid(), f"Invalid encoder {v} - check that it is properly versioned"
				encoders[v.getVersion()] = v
		return encoders

	@classmethod
	def latestEncoderVersion(cls)->int:
		"""Return the latest encoder class version.
		"""
		return max(cls.encoderVersionMap().keys())

	@classmethod
	def getEncoder(cls, versionIndex:int=None)->T.Type[EncoderBase]:
		"""by default, retrieve the latest"""
		if versionIndex is None:
			return cls.encoderVersionMap()[max(cls.encoderVersionMap().keys())]
		return cls.encoderVersionMap()[versionIndex]


	@classmethod
	def checkIsValid(cls)->bool:
		"""Check that the class has been defined correctly.
		"""
		if cls.encoderBaseCls is None:
			raise ValidationError(f"Encoder base class not set for {cls}")
		if not isinstance(cls.uniqueAdapterName, str) :
			raise ValidationError(f"Unique adapter name not set for {cls}")
		if not issubclass(cls.encoderBaseCls, EncoderBase):
			raise ValidationError(f"Encoder base class {cls.encoderBaseCls} is not a subclass of {EncoderBase}")
		if not cls.encoderVersionMap():
			raise ValidationError(f"No encoders defined for {cls}, no versionMap derived")
		return True

	# main methods
	@classmethod
	def encode(cls, obj, encoderVersion:int=None)->dict:
		"""Encode the object into a dict - if no version is specified,
		use the latest. (Latest should probably always be used when saving).

		mixing in the key for format data sets off my spidey-sense, but the alternative
		is creating a new container level of dict for each new type in a hierarchy,
		which gets tedious to read.
		"""
		encoder = cls.getEncoder(versionIndex=encoderVersion)
		return {
			**encoder.encode(obj),
			FORMAT_DATA_KEY : cls.getFormatData(encoder.getVersion())
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
		serialType = coderef.resolveCodeRef(formatData[cls.VERSION_DATA_TYPE_KEY])

		# decode
		return encoder.decode(
			serialType, serialData[ENCODE_DATA_KEY])
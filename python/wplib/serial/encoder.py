
from __future__ import annotations
import typing as T


from wplib.validation import ValidationError

def version(index:int):
	"""Decorator for classes - adjusts class name,
	sets class attribute.
	"""
	assert index > 0, f"Version index must be greater than 0, not {index}"
	def _inner(cls:type[EncoderBase]):
		cls.__name__ = f"{cls.__name__}_V{index}"
		cls._versionIndex = index
		return cls
	return _inner



class EncoderBase:
	"""Base class for encoding and decoding types.
	Subclass this and define the type to encode.

	Decorate real classes with the version decorator,
	don't set attributes in definition.
	"""

	# PROCEDURAL CONTROL - DO NOT EDIT BY HAND
	_versionIndex:int = -1
	# END CONTROL

	encodeType:T.Type = object

	versionDec = version

	@classmethod
	def encode(cls, obj:encodeType)->dict:
		raise NotImplementedError

	@classmethod
	def decode(cls, serialCls:encodeType, serialData:dict)->encodeType:
		raise NotImplementedError


	@classmethod
	def checkIsValid(cls)->bool:
		"""Check that the class has been defined correctly.
		:raises ValidationError: if invalid
		"""
		if cls._versionIndex == -1: # not properly versioned
			raise ValidationError(f"Encoder {cls} is not properly versioned")
		return True

	@classmethod
	def getVersion(cls)->int:
		return cls._versionIndex


"""version of an encoder class should know its type, and that's it"""


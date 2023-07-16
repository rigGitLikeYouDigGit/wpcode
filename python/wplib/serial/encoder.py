
from __future__ import annotations
import typing as T

from enum import Enum

def _enumSerialData(obj:Enum):
	return obj.value
def _enumDeserialise(cls:T.Type[Enum], serialData):
	return cls._value2member_map_[serialData]

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

	@classmethod
	def encode(cls, obj:encodeType)->dict:
		raise NotImplementedError

	@classmethod
	def decode(cls, serialData:dict)->encodeType:
		raise NotImplementedError

	@classmethod
	def _defineVersion(cls:T.Type, index:int):
		"""Decorator for classes - adjusts class name,
		sets class attribute.
		"""
		assert index not in cls._versionMap, f"Duplicate version defined for index index {index} for {cls}"
		cls.__name__ = f"{cls.__name__}_V{index}"
		cls.versionIndex = index

def version(index:int):
	"""Decorator for classes - adjusts class name,
	sets class attribute.
	"""
	def _inner(cls):
		cls.__name__ = f"{cls.__name__}_V{index}"
		cls.versionIndex = index
		return cls
	return _inner


"""version of an encoder class should know its type, and that's it"""

class SerialAdapter:
	"""Helper class to be used with external types,
	defining our own rules for saving and loading.

	This outer type should also define the type's serial-UID,
	and hold all versioned encoders.
	"""



class EnumEncoderBase(EncoderBase):
	"""Base class for encoding and decoding Enum types.
	Subclass this and define the enum type to encode.
	"""
	cls:T.Type[Enum] = Enum

@version(1)
class EnumEncoder:
	@classmethod
	def encode(cls, obj:Enum):
		return _enumSerialData(obj)
	@classmethod
	def decode(cls, serialData):
		return _enumDeserialise(cls.cls, serialData)


print(EnumEncoder.versionIndex)
print(EnumEncoder.__name__)
print(EnumEncoder)



from __future__ import annotations
import typing as T

from enum import Enum

from wplib.serial.encoder import EncoderBase, version
from wplib.serial.adaptor import SerialAdaptor

from wplib.serial.register import SerialRegister

def _enumSerialData(obj:Enum):
	return obj.value
def _enumDeserialise(cls:T.Type[Enum], serialData):
	return

class EnumEncoderBase(EncoderBase):
	"""Base class for encoding and decoding Enum types.
	Subclass this and define the enum type to encode.
	"""
	encodeType = Enum

class EnumAdaptor(SerialAdaptor):
	encoderBaseCls = EnumEncoderBase

	uniqueAdapterName = "stdEnum"

	@version(1)
	class EnumEncoder(EnumEncoderBase):
		@classmethod
		def encode(cls, obj: Enum):
			return obj.value

		@classmethod
		def decode(cls, serialCls:type[Enum], serialData:dict) ->Enum:
			return serialCls._value2member_map_[serialData]


SerialRegister.register(EnumAdaptor)




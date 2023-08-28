

from __future__ import annotations
import typing as T

from enum import Enum
from wplib import CodeRef
from wplib.serial.encoder import EncoderBase, version
from wplib.serial.adaptor import SerialAdaptor

from wplib.serial.register import SerialRegister


class TypeEncoderBase(EncoderBase):
	"""Base class for encoding and decoding Enum types.
	Subclass this and define the enum type to encode.
	"""
	encodeType = type

class TypeAdaptor(SerialAdaptor):
	encoderBaseCls = TypeEncoderBase

	uniqueAdapterName = "stdEnum"

	@version(1)
	class Encoder(EncoderBase):
		@classmethod
		def encode(cls, obj: type):
			return {"data" : CodeRef.get(obj)}

		@classmethod
		def decode(cls, serialCls:type, serialData:dict) ->Enum:
			return CodeRef.resolve(serialData["data"])


SerialRegister.registerAdaptor(TypeAdaptor)






from __future__ import annotations
import typing as T

from enum import Enum
from wplib import CodeRef
from wplib.serial.adaptor import SerialAdaptor, EncoderBase

from wplib.serial.register import SerialRegister




class TypeAdaptor(SerialAdaptor):

	uniqueAdapterName = "type"

	@classmethod
	def serialType(cls) ->type:
		return type

	@SerialAdaptor.encoderVersion(1)
	class Encoder(EncoderBase):
		@classmethod
		def encodeObject(cls, obj: type):
			return {"data" : CodeRef.get(obj)}

		@classmethod
		def decodeObject(cls, serialCls:type, serialData:dict) ->Enum:
			return CodeRef.resolve(serialData["data"])


SerialRegister.registerAdaptor(TypeAdaptor)




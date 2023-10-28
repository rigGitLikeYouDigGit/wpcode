

from __future__ import annotations
import typing as T

from enum import Enum

from wplib.serial.adaptor import SerialAdaptor, EncoderBase

from wplib.serial.main import SerialRegister

def _enumSerialData(obj:Enum):
	return obj.value
def _enumDeserialise(cls:T.Type[Enum], serialData):
	return

class EnumAdaptor(SerialAdaptor):

	uniqueAdapterName = "stdEnum"

	@classmethod
	def serialType(cls) ->type:
		return Enum

	@SerialAdaptor.encoderVersion(1)
	class Encoder(EncoderBase):
		@classmethod
		def encodeObject(cls, obj: Enum,  encodeParams:dict):
			return {"data" : obj.value}

		@classmethod
		def decodeObject(cls, serialCls:type[Enum], serialData:dict,
		                 decodeParams:dict) ->Enum:
			return serialCls._value2member_map_[serialData["data"]]


SerialRegister.registerAdaptor(EnumAdaptor)






from __future__ import annotations
import typing as T

from enum import Enum
from wplib import CodeRef
from wplib.serial.main import SerialAdaptor, SerialRegister




class TypeAdaptor(SerialAdaptor):

	uniqueAdapterName = "type"

	@classmethod
	def serialType(cls) ->type:
		return type

	# @SerialAdaptor.encoderVersion(1)
	# class Encoder(EncoderBase):
	@classmethod
	def _encodeObject(cls, obj: type,  encodeParams:dict):
		return {"data" : CodeRef.get(obj)}

	@classmethod
	def _decodeObject(cls, serialCls:type, serialData:dict,
	                 decodeParams:dict, formatVersion=-1) ->Enum:
		return CodeRef.resolve(serialData["data"])


SerialRegister.registerAdaptor(TypeAdaptor)




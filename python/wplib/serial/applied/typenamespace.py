

from __future__ import annotations
import typing as T


from wplib.serial.adaptor import SerialAdaptor, EncoderBase

from wplib.serial.register import SerialRegister

from wplib.object.namespace import TypeNamespace


class TypeNamespaceAdaptor(SerialAdaptor):

	uniqueAdapterName = "typeNamespace"

	@SerialAdaptor.encoderVersion(1)
	class Encoder(EncoderBase):
		@classmethod
		def encodeObject(cls, obj: TypeNamespace.base(),  encodeParams:dict):
			return {"data" : obj.clsName()}

		@classmethod
		def decodeObject(cls, serialCls:type[TypeNamespace], serialData:dict,
		                 decodeParams:dict) ->TypeNamespace:
			return serialCls.members()[serialData["data"]]


SerialRegister.registerAdaptor(TypeNamespaceAdaptor)




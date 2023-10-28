

from __future__ import annotations
import typing as T


from wplib.serial.main import SerialAdaptor, SerialRegister

from wplib.object.namespace import TypeNamespace


class TypeNamespaceAdaptor(SerialAdaptor):

	uniqueAdapterName = "typeNamespace"

	# @SerialAdaptor.encoderVersion(1)
	# class Encoder(EncoderBase):
	@classmethod
	def _encodeObject(cls, obj: TypeNamespace.base(),  encodeParams:dict):
		return {"data" : obj.clsName()}

	@classmethod
	def _decodeObject(cls, serialCls:type[TypeNamespace], serialData:dict,
	                 decodeParams:dict, formatVersion=-1) ->TypeNamespace:
		return serialCls.members()[serialData["data"]]


SerialRegister.registerAdaptor(TypeNamespaceAdaptor)




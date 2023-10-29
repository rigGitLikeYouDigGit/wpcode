

from __future__ import annotations
import typing as T


from wplib.serial.main import SerialAdaptor, SerialRegister

from wplib.object.namespace import TypeNamespace

from wplib.coderef import CodeRef


class TypeNamespaceAdaptor(SerialAdaptor):

	uniqueAdapterName = "typeNamespace"

	@classmethod
	def serialType(cls) ->type:
		return TypeNamespace

	@classmethod
	def encode(cls, obj, encodeParams:dict=None) ->dict:
		codeStr = CodeRef.get(obj)

		return "T:"+codeStr

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

#raise


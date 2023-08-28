

from __future__ import annotations
import typing as T


from wplib.serial.encoder import EncoderBase, version
from wplib.serial.adaptor import SerialAdaptor

from wplib.serial.register import SerialRegister

from wplib.object.namespace import TypeNamespace

def _enumSerialData(obj:Enum):
	return obj.value
def _enumDeserialise(cls:T.Type[Enum], serialData):
	return

class TypeNamespaceEncoderBase(EncoderBase):
	"""Serialise and retrieve namespace types.
	"""
	encodeType = TypeNamespace

class TypeNamespaceAdaptor(SerialAdaptor):
	encoderBaseCls = TypeNamespaceEncoderBase

	uniqueAdapterName = "typeNamespace"

	@version(1)
	class Encoder(TypeNamespaceEncoderBase):
		@classmethod
		def encode(cls, obj: TypeNamespace.base()):
			return {"data" : obj.clsName()}

		@classmethod
		def decode(cls, serialCls:type[TypeNamespace], serialData:dict) ->TypeNamespace:
			return serialCls.members()[serialData["data"]]


SerialRegister.registerAdaptor(TypeNamespaceAdaptor)




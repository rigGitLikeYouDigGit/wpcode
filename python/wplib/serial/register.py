from __future__ import annotations
import typing as T

from wplib import inheritance, CodeRef
from wplib.serial.adaptor import SerialAdaptor
from wplib.serial.constant import FORMAT_DATA_KEY
from wplib.serial import lib


"""global register for adaptor classes - top-level
interface for serialisation system.

Adaptors must be registered explicitly - they also must not
collide, only one adaptor per class.
"""

class SerialRegister:

	# map of { serial type : adaptor for that type }
	typeAdaptorMap = {} # type: T.Dict[T.Type, SerialAdaptor]

	@classmethod
	def registerAdaptor(cls, adaptor:T.Type[SerialAdaptor]):
		adaptor.checkIsValid(), f"Adaptor {adaptor} is not valid"

		items = set(cls.typeAdaptorMap.items())

		if (adaptor.serialType(), adaptor) in items: # exact pair already registered
			return

		assert adaptor.serialType() not in cls.typeAdaptorMap, f"Adaptor {adaptor} already registered"

		cls.typeAdaptorMap[adaptor.serialType()] = adaptor

	@classmethod
	def adaptorForClass(cls, forCls:T.Type)->SerialAdaptor:
		"""Get the adaptor for the given class.
		"""
		return inheritance.superClassLookup(cls.typeAdaptorMap, forCls, default=None)

	# @classmethod
	# def adaptorForCodeRefString(cls, codeRefString:str)->SerialAdaptor:
	# 	"""Get the adaptor for the given code ref string.
	# 	We rely on the code ref still being valid here - improve later if needed.
	# 	Maybe there's a benefit to registering adaptors alongside their own
	# 	code ref paths, so the class has a record of previous places it was defined
	# 	"""
	# 	return cls.adaptorForClass(CodeRef.resolve(codeRefString))

	@classmethod
	def adaptorForData(cls, data:dict)->SerialAdaptor:
		"""Get the adaptor for the given data dict.
		"""
		return cls.adaptorForClass(CodeRef.resolve(lib.getDataCodeRefStr(data)))

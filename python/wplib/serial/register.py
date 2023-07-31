from __future__ import annotations
import typing as T

from wplib.serial.adaptor import SerialAdaptor

"""global register for adaptor classes - top-level
interface for serialisation system.

Adaptors must be registered explicitly - they also must not
collide, only one adaptor per class.
"""

class SerialRegister:

	# map of { serial type : adaptor for that type }
	typeAdaptorMap = {} # type: T.Dict[T.Type, SerialAdaptor]

	@classmethod
	def register(cls, adaptor:T.Type[SerialAdaptor]):
		adaptor.checkIsValid(), f"Adaptor {adaptor} is not valid"

		items = set(cls.typeAdaptorMap.items())

		if (adaptor.serialType(), adaptor) in items: # exact pair already registered
			return

		assert adaptor.serialType() not in cls.typeAdaptorMap, f"Adaptor {adaptor} already registered"

		cls.typeAdaptorMap[adaptor.serialType()] = adaptor


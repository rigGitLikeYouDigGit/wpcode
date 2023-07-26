
from __future__ import annotations

import typing as T

class ClassHookMeta(type):
	"""provides hooks at various points of class declaration"""

	@classmethod
	def preClassNew(mcs, *newArgs, **newKwargs)->tuple[tuple, dict]:
		"""called before the __new__ function is called on a newly declared class
		use to filter arguments to new class"""
		return newArgs, newKwargs

	@classmethod
	def postClassNew(mcs, newCls, *newArgs, **newKwargs):
		"""called after the __new__ function is called on a newly declared class
		returns the newCls
		"""
		try:
			newCls = newCls.onPostMetaNew(newCls, *newArgs, **newKwargs)
		except KeyError:
			pass
		return newCls

	def __new__(mcs, *args, **kwargs):
		args, kwargs = mcs.preClassNew(*args, **kwargs)
		newCls = type.__new__(mcs, *args, **kwargs)
		newCls = mcs.postClassNew(newCls, *args, **kwargs)
		return newCls


class ClassHookTemplate(metaclass=ClassHookMeta):
	"""a template class providing integration with class hook methods
	either inherit from this, or inherit from the metaclass above -
	not sure which is easier, but this seems better"""

	@staticmethod
	def onPostMetaNew(newCls, *newArgs, **newKwargs)->T.Type:
		return newCls




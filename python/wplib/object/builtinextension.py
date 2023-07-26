
from __future__ import annotations

"""extending sets, lists, dicts etc"""

import typing as T

from wplib.object.classhookmeta import ClassHookTemplate


class PostDecMethodWrapper(ClassHookTemplate):
	"""wraps the result of any direct methods
	to given types
	could even break this down further into a general abstract
	method wrapper
	"""

	# use None to denote returning type of this class itself
	fnNameTypeMap : dict[tuple[str], callable] = {}

	@classmethod
	def _wrapMethods(cls):
		def wrapMethodClosure(name, resultType):
			def inner(self, *args):
				result = getattr(super(cls, self), name)(*args)
				wrapType = resultType or cls
				return wrapType(result)
			inner.fn_name = name
			setattr(cls, name, inner)

		for fnNames, destType in cls.fnNameTypeMap.items():
			if destType is None: destType = cls
			for name in fnNames:
				wrapMethodClosure(name, resultType=destType)

	@staticmethod
	def onPostMetaNew(newCls:T.Type[PostDecMethodWrapper], *newArgs, **newKwargs) ->T.Type:
		newCls._wrapMethods()
		return newCls


class UserSet(set, PostDecMethodWrapper):

	fnNameTypeMap = {
		('__ror__', 'difference_update', '__isub__',
		 'symmetric_difference', '__rsub__', '__and__', '__rand__', 'intersection',
		 'difference', '__iand__', 'union', '__ixor__',
		 'symmetric_difference_update', '__or__', 'copy', '__rxor__',
		 'intersection_update', '__xor__', '__ior__', '__sub__',
		 ) : None,
	}


if __name__ == '__main__':
	a = [4, 5, 6]
	b = [6, 7, 8]
	usa = UserSet(a)
	usb = UserSet(b)

	print(usa, type(usa))
	print(usa - usb)
	print(usa - set(b))
	print(set(a) - usb)



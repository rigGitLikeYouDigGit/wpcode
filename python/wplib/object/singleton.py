
from __future__ import annotations
import typing as T


class SingletonDecorator:
	"""
	properly type hinting class decorators is difficult,
	couldn't find a good way to type hint
	returning a call on the given type, when we don't know the
	type

	so instead we just have the TYPE_CHECKING override,
	and we have to use __call__ on the decorator, which is a bit less
	explicit than .instance() or something

	"""

	def __init__(self, decorated):
		self._decorated = decorated
		self._instance = None

	def __call__(self):
		if self._instance is None:
			self._instance = self._decorated()
		return self._instance

	def cls(self)->type:
		return self._decorated

	def __instancecheck__(self, inst):
		return isinstance(inst, self._decorated)

if T.TYPE_CHECKING:
	SingletonDecorator = lambda x : x


class SingletonMeta:
	"""TODO if needed"""
	pass

if __name__ == '__main__':

	@SingletonDecorator
	class _SingletonTest:
		"""A demonstration of the Singleton pattern"""
		def __init__(self):
			print('Creating singleton')

		def instanceFn(self):
			print('instance function called')

		@classmethod
		def clsFn(cls):
			print('class function called')


	s1 = _SingletonTest().instanceFn()
	s2 = _SingletonTest().instanceFn()





from __future__ import annotations
import typing as T

"""any methods of achieving a post-init call,
running after the full init of an object, regardless"""

class PostInitBase:
	"""all implementations of PostInit should run the __postInit__ method
	after the init of the object, regardless of how it is called
	"""
	def __postInit__(self):
		"""run after the init of this object has fully resolved"""


class PostInitCreateMixin(PostInitBase):
	""" the absolute simplest way,
	but probably conflicts with other class
	"create" methods
	"""
	@classmethod
	def create(cls, *args, **kwargs):
		"""create instance, then run post init"""
		ins = cls(*args, **kwargs)
		ins.__postInit__()
		return ins


def postInitWrap(cls):
	"""wrap class with post init"""

	cls.__oldInit__ = cls.__init__
	def _patchInit(self, *args, **kwargs):
		cls.__oldInit__(self, *args, **kwargs)
		self.__postInit__()
	cls.__init__ = _patchInit

	return cls



if __name__ == '__main__':
	@postInitWrap
	class TestCls(PostInitBase):
		def __init__(self, *args, **kwargs):
			print('init', args, kwargs)

		def __postInit__(self):
			print('post init', self)



	t = TestCls("a", "b", c="c")


from __future__ import annotations
import typing as T
from wplib import log
from dataclasses import dataclass

from wplib import inheritance

"""any methods of achieving a post-init call,
running after the full init of an object, regardless"""

@dataclass
class DataTest:
	s : str

	def __post_init__(self):
		pass

sd = DataTest("hi")


class PostInitBase:
	"""all implementations of PostInit should run the __postInit__ method
	after the init of the object, regardless of how it is called
	"""
	def __post_init__(self, *args, **kwargs):
		"""run after the init of this object has fully resolved"""

class PostInitMeta(type):

	# @classmethod
	# def __prepare__(metacls, name, bases):
	# 	from wplib import log
	# 	#log("postinit prepare", metacls, name, bases)
	# 	return {}

	def __call__(cls, *args, **kwargs):
		# log("postInit", cls, args, kwargs)
		# log("cls mro", cls.__mro__)
		# log("type mro", type(cls).__mro__)
		# t, method = inheritance.superLookup(type(cls), "__call__")
		# log("t method", t, method)
		# if t is None: # no other __call__ found, revert to type
		# 	obj = type.__call__(cls)
		# else:
		# 	#obj = inheritance.clsSuper(type(cls)).__call__(*args, **kwargs)
		# 	obj = method.__call__(type(cls), *args, **kwargs)
		# # try: #TODO: work out how to pack in arguments here
		# # 	obj = method.__call__(*args, **kwargs)
		# # except TypeError:
		# # 	obj = method.__call__()
		try:
			obj = super(PostInitMeta, cls).__call__( *args, **kwargs)
		except TypeError:
			obj = inheritance.clsSuper(type(cls)).__call__(cls, *args, **kwargs)

		if hasattr(obj, "__post_init__"):
			obj.__post_init__(*args, **kwargs)
		return obj


class PostInitCreateMixin(PostInitBase):
	""" the absolute simplest way,
	but probably conflicts with other class
	"create" methods
	"""
	@classmethod
	def create(cls, *args, **kwargs):
		"""create instance, then run post init"""
		ins = cls(*args, **kwargs)
		ins.__post_init__(*args, **kwargs)
		return ins


def postInitWrap(cls):
	"""wrap class with post init -
	avoid unless absolutely necessary, since this
	will daisy-chain each declared post_init
	function if you use it multiple times in inheritance"""

	cls.__old_init__ = cls.__init__
	def _patchInit(self, *args, **kwargs):
		cls.__old_init__(self, *args, **kwargs)
		self.__post_init__(*args, **kwargs)
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

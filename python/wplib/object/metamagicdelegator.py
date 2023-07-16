
from __future__ import annotations

"""how much complexity is too much?
today we will answer this question
or the search will go on"""


def magicMethodLookupPatch(cls, metacls, classMethodName: str, magicMethodName:str):
	"""delegate """
	def _delegateToClsMagicMethod(cls, *args, **kwargs):
		#print("delegate to ", magicMethodName, args, kwargs)
		try:
			return getattr(cls, classMethodName)(*args, **kwargs)
		except AttributeError:
			return getattr(type, magicMethodName)(cls, *args, **kwargs)

	return _delegateToClsMagicMethod


class _MetaMagicDelegator(type):
	"""metaclass for delegating metaclass-level operators
	to class-level classmethods

	Need to redo this at some point - this is fine, but declaring
	the len and bool-relevant methods in base class makes it a chore to
	reimplement them, since suddenly the class evaluates as False
	if it has no length.

	Better system would look over class dict at definition time,
	detect "__class_" prefix and patch them in dynamically.

	We can't look up methods lazily when not found, because things like
	adding don't delegate to __getattr__ - they just fail. Have to be proactive
	BUT if we patch them on directly, then 2 classes with the same metaclass but
	different methods will clash.

	Need to set methods on metaclass to delegated lookup, back to the class
	in question, THEN call the class methods.

	"""

	@staticmethod
	def patchMetaClsMethodsfromCls(cls, metacls):
		"""look over class dict for properly named methods -
		patch them on to metaclass.
		"""
		#print("patching", cls, metacls)
		for key, value in cls.__dict__.items():
			if key.startswith("__class_") and (
					callable(value) or isinstance(value, (staticmethod, classmethod))):
				methodName = "__" + key.split("__class_")[1]
				# if methodName == "__repr__": # careful with repr, as it can cause infinite recursion
				# 	continue
				setattr(metacls, methodName,
				        magicMethodLookupPatch(cls,
				                               metacls,
				                               key,
				                               methodName)
				        )

	def __new__(cls, *args, **kwargs):
		cls = type.__new__(cls, *args, **kwargs)
		cls.patchMetaClsMethodsfromCls(cls, type(cls))
		return cls

	# def __call__(cls, *args, **kwargs):
	# 	cls.patchMetaClsMethodsfromCls(cls, type(cls))
	# 	return cls


class ClassMagicMethodMixin(object,
							metaclass=_MetaMagicDelegator):
	"""Main mixin class to override metaclass-level magic methods.

	A classmethod defined as follows:
	>>>@classmethod
	>>>def __class_methodname__(cls, *args, **kwargs):

	will be called on the class itself by the metaclass, when needed.

	For example,

	>>>class MyClass(ClassMagicMethodMixin):
	>>>	@classmethod
	>>>	def __class_contains__(cls):
	>>>     assert cls is ClassMagicMethodMixin
	>>>		return True


	"""

	# # actual metaclass methods, used to avoid redefining a new metaclass
	@classmethod
	def __class_instancecheck__(cls, other)->bool:
		#print("instance check", cls, other)
		return type.__instancecheck__(cls, other)
	# #
	# @classmethod
	# def __class_add__(cls, other):
	# 	print("add", cls, other)
	# 	return 1 + other
	#
	# @classmethod
	# def __class_getitem__(cls, item):
	# 	return item + "adaa"


if __name__ == '__main__':

	instance = ClassMagicMethodMixin()

	print(isinstance(instance, ClassMagicMethodMixin))
	print(isinstance(ClassMagicMethodMixin, _MetaMagicDelegator))

	# result = ClassMagicMethodMixin["test"]
	# print(result)
	# #
	# print(ClassMagicMethodMixin + 3)

	#
	# ClassMagicMethodMixin["gg"] = 3
	#
	# print(len(ClassMagicMethodMixin))




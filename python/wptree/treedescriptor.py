
from __future__ import annotations
"""descriptor allowing basic interaction with tree auxProperties
without separate calls"""

import typing as T
from wplib.sentinel import Sentinel
if T.TYPE_CHECKING:
	from wptree import TreeInterface

class TreeBranchDescriptor:
	"""allowing object dot access to specific branches,
	for predefined data formats"""
	def __init__(self, key:TreeInterface.keyT, create=True,
	             useValue=False):
		"""if useValue, will get / set the value of the target branch -
		otherwise just returns the branch"""
		self.key = key
		self.create = create
		self.useValue = useValue

	def __get__(self, instance:TreeInterface, owner)->(TreeInterface, object):
		branch = instance(self.key, create=self.create)
		return branch.value if self.useValue else branch

	def __set__(self, instance:TreeInterface, value:(TreeInterface, object)):
		"""if not useValue, check that value is a valid branch to replace
		the existing"""
		if not self.useValue:
			from tree import TreeInterface
			assert isinstance(value, TreeInterface), "Replacement branch given to descriptor must be a tree or tree subclass"

			parentBranch = instance(self.key[:-1], create=self.create)
			parentBranch.addBranch(value, force=True)
			return

		branch = instance(self.key, create=self.create)
		branch.value = value

class TreePropertyDescriptor(object):
	"""
	Define consistent tree aux auxProperties via class descriptors -
	These do not store their own value, only interact with tree
	aux property maps
	"""

	def templateDefaultFunction(instance:TreeInterface,
	                            key :str):
		"""demo function for a descriptor default lambda"""
		raise NotImplementedError

	def __init__(
			self,
			key :str,
			default:(T.Callable[[TreeInterface, TreeInterface.keyT], object],
			         object)=None,
			inherited=False,
			defaultSetsValue=False,
			#returnBranch=False,
			desc="",
		):
		"""
		:param key: aux property key to use
		:param default: default object or function to evaluate if property is not set on tree
		:param inherited:
			if true, will look up tree inheritance for the property;
			if false, only look up on direct parent tree
		:param defaultSetsValue:
			if true, getting an empty property will set the value to the default on the tree
		:param desc: description of property
		"""
		self.key = key
		self.default = default
		self.inherited = inherited
		self.defaultSetsValue = defaultSetsValue
		#self.returnBranch = returnBranch
		self.desc = desc

	def __get__(self, instance :TreeInterface, owner):
		"""look up property on given tree """
		if self.inherited:
			result = instance.getInherited(
				self.key,
				returnBranch=False,
				default=Sentinel.FailToFind
			)
		else:
			result = instance.getAuxProperty(
				self.key, default=Sentinel.FailToFind)
		if result is Sentinel.FailToFind:
			result = self._evalDefault(instance, self.key)

			# optionally set the value on this branch to the default result
			if self.defaultSetsValue:
				instance.setAuxProperty(self.key, result)
		return result

	def __set__(self, instance :TreeInterface, value):
		# do not set value if equal to default
		if value == self._evalDefault(instance, self.key):
			return
		instance.setAuxProperty(self.key, value)

	def _evalDefault(self, instance :TreeInterface, key :str):
		"""if default is a callable, pass it the lookup tree object and key
		once arg trimming works"""
		if callable(self.default):
			return self.default(instance, key)
		return self.default

class TreeValueDescriptor:
	"""represents a key in value dictionary -
	"""

	def __init__(self, key):
		self.key = key

	def __get__(self, instance:TreeInterface, owner):
		return instance.getValue()[self.key]

	def __set__(self, instance:TreeInterface, value):
		instance.getValue()[self.key] = value
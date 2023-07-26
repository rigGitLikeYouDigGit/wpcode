from __future__ import annotations
import typing as T


from dataclasses import dataclass

import wplib.sentinel
from wptree import Tree

from wp import constant
from wp.object.node import DepNode


"""Commonly in hierarchies and nested systems, a data point
must control some attributes of all objects below it,
unless a lower object itself provides another override.

Tree has a method of this with the inherited auxProperties system, but
here we define a more explicit and extensible way.

Parallel class inheritance for normal data - 
an override may draw from multiple sources, and in this case
defines an override resolution order.


vocab:
"immediate" - only concerning single node in question, no interaction
with hierarchy or dependency
"aggregate" - query dependency to find result of override system


does not interact with deltas or dirty nodes yet
complexity really is a drug

"""
addressT = T.Union[str, T.Sequence[str]]


# @dataclass
# class OverrideDefinition:
# 	"""defines a single real override value on an override node"""
# 	name:str
# 	value:T.Any
#
# 	"""if forceDefine, override will be defined even if parent
# 	value matches it"""
# 	forceDefine:bool = False
#
# 	"""if sticky, override will persist if parent value changes
# 	to match this override - if not, it will be lost"""
# 	sticky:bool = False


class OverrideEntry:
	"""single override entry in a node's override system.
	doesn't do ANYTHING active, replace with dataclass.

	Resolving method itself is overridden, available for control injection,
	etc.

	Consider different typed classes of override to give some hint to compositing
	functions.

	Returns new OverrideEntry object
	"""
	def __init__(self, value:T.Any):
		self.value = value

	def __repr__(self):
		return f"OverrideEntry({self.value})"


def applyImmediateOverride(
		immediateOwner:Overrider,
		axis:str,
		existingOverride:OverrideEntry,
		immediateOverride:OverrideEntry,

)->OverrideEntry:
	""" Default function to apply overrides - this function itself should be placed
	in the override system, and can be specified to change behaviour.

	apply immediate override to existing override, returning new
	override entry"""
	return OverrideEntry(immediateOverride.value)

class OverrideProperty:
	"""for defining and accessing overrides directly.

	"""

	def templateDefaultFunction(instance:NodeOverrider,
	                            address:addressT,
	                            ):
		"""demo function for a descriptor default lambda"""
		raise NotImplementedError

	def __init__(self, address:(str, T.Sequence[str]),
	             default:[T.Any, T.Callable[[NodeOverrider, addressT], T.Any]]= wplib.constant.Sentinel.Empty,
	             ):
		self.address = address

		# we need to check if the default value of the default is default
		self.default = default

	def __get__(self, instance :NodeOverrider, owner):
		"""look up property on given overrider """
		result = instance.getOverride(self.address,
		                              fallback=wplib.constant.Sentinel.FailToFind)
		if result is not wplib.constant.Sentinel.FailToFind:
			return result
		if self.default is wplib.constant.Sentinel.Empty:
			raise KeyError(f"no override found for {self.address}")
		defaultResult = self._evalDefault(instance, self.address)
		return defaultResult

	def __set__(self, instance :NodeOverrider, value):
		# do not set value if equal to default
		# if value == self._evalDefault(instance, self.address):
		# 	return
		instance.setOverride(self.address, value)

	def _evalDefault(self, instance :NodeOverrider, key :str):
		"""if default is a callable, pass it the lookup overrider and address
		"""
		if callable(self.default):
			return self.default(instance, key)
		return self.default


class OverrideReservedKeys:
	"""look these up directly without composing tree.
	if these are used, they are directly replaced in tree,
	overriding active behaviour"""
	resolveFn = "resolveFn" # retrieve function to use, to resolve other override values

class Overrider:
	"""An overrider is the end / intersection of at least one
	named axis of inheritance.

	At any point in any of these axes, a parent may define an override
	of a named attribute - override here simply meaning that parent has input to
	the final result of that attribute. It may replace the old value, add it,
	etc - resolving that override with the previous is delegated.

	Overridden attributes are indexed by tree.


	Generate first lightweight tree to get functions to resolve overrides and
	any other active behaviour

	Then do any data compositing

	"""

	resolveFnKey = "resolveFn"

	def __init__(self):
		"""Define empty override tree"""
		self._overrideTree :Tree[str, T.Any] = Tree("root")

	def getParentOverrider(self, forAxis="main")->Overrider:
		"""get parent override object"""
		raise NotImplementedError



	def defineImmediateOverride(self, address:addressT, value:T.Any):
		"""define override at given address"""
		self._overrideTree[address] = value

	def getReservedOverrides(self)->dict[str, T.Any]:
		"""get overrides for reserved keys, like functions used to collate
		more complex values"""






		return self._overrideTree.get(OverrideReservedKeys.resolveFn, {})


class NodeOverrider(DepNode):
	"""class for objects that can override other objects.
	differentiating input map keys is just done through string
	prefixes for now - maybe this is enough.

	Most cases will probably inherit from UidElement too,
	but that is left to caller.

	Tree is used as base override structure, allowing arbitrarily
	nested overrides
	"""

	OverrideProperty = OverrideProperty

	def __hash__(self):
		return id(self)

	@classmethod
	def inputMapKeyPrefix(cls) -> str:
		"""return prefix for override keys in input map"""
		return "override_"

	def __init__(self):
		DepNode.__init__(self)
		self._overrideTree :Tree[str, T.Any] = Tree("root")

	def getOverrideInputs(self)->dict[str, NodeOverrider]:
		"""get override input nodes"""
		return {k : v for k, v in self.getInputs().items() if k.startswith(self.inputMapKeyPrefix())}

	def immediateOverrideResolutionOrder(self)->list[NodeOverrider]:
		"""return list of override input nodes in order of resolution.
		only considers immediate overrides, not aggregate"""
		return [self] + list(self.getOverrideInputs().values())

	def aggregateOverrideBases(self, _seen:set[NodeOverrider]=None)->T.Iterator[NodeOverrider]:
		"""return total list of override input nodes in order of resolution.
		"""
		_seen = _seen or set()
		for node in self.immediateOverrideResolutionOrder():
			if node not in _seen:
				_seen.add(node)
				yield node
				yield from node.aggregateOverrideBases(_seen)



	def setOverrideInput(self, node:NodeOverrider, inputName="main"):
		"""set override input node"""
		self.setInputNode(self.inputMapKeyPrefix() + inputName, node)

	def getOverrideInput(self, inputName="main")->NodeOverrider:
		"""get override input node"""
		return self.getInput(self.inputMapKeyPrefix() + inputName)


	# setting and getting overrides
	def setOverride(self, address:addressT, value: T.Any, forceDefine: bool = False, sticky: bool = False):
		"""add override definition"""
		self._overrideTree(address, create=True).setValue(value)
	def removeOverride(self, address:addressT):
		"""remove override definition"""
		self._overrideTree.remove(address)

	def getImmediateOverrideTree(self)->Tree:
		"""get immediate override map"""
		return self._overrideTree

	def getImmediateOverride(self, address:addressT,
	                         fallback=None)->T.Any:
		"""get immediate override value"""
		return self.getImmediateOverrideTree().get(address, fallback)

	def getAggregateOverrideTree(self)->Tree:
		"""get aggregate override tree -
		sequentially override by all branches appearing
		in each node's tree"""
		tree = Tree("root")
		for node in self.aggregateOverrideBases():
			for branch in node.getAggregateOverrideTree():
				# if branch is already defined (overridden), don't override
				if not tree.getBranch(branch.address()):
					tree(branch.address(), create=True).value = branch.value
		return tree


	def getOverride(self, address:addressT, fallback=None)->T.Any:
		"""get aggregate override value - we assume
		most calls will want full aggregate effect"""
		for node in self.aggregateOverrideBases():
			result = node.getImmediateOverride(
				address, fallback=wplib.constant.Sentinel.FailToFind)
			if result is not wplib.constant.Sentinel.FailToFind:
				return result
		return fallback


if __name__ == '__main__':

	# test override
	class TestOverrider(NodeOverrider):
		overrideA = NodeOverrider.OverrideProperty("a", default=1)


	rootOverride = TestOverrider()

	branchOverride = TestOverrider()
	branchOverride.setOverrideInput(rootOverride)

	leafOverride = TestOverrider()
	leafOverride.setOverrideInput(branchOverride)

	assert leafOverride.overrideA == 1

	rootOverride.overrideA = "gg"

	assert leafOverride.overrideA == "gg"





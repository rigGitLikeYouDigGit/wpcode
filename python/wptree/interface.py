from __future__ import annotations
import typing as T

import pprint, copy

from dataclasses import dataclass
from collections import namedtuple

from wplib.sequence import flatten, resolveSeqIndex
from wplib.object import Signal, EventDispatcher#, EventBase, DeepVisitor
from wplib import TypeNamespace, log
from wplib.sentinel import Sentinel
from wplib.wpstring import incrementName
from wplib import CodeRef

from wplib.object import VisitAdaptor, Visitable
from wplib.serial import Serialisable, SerialAdaptor
from wplib.pathable import Pathable


#from wptree.delta import TreeDeltas
from wptree.treedescriptor import TreePropertyDescriptor

"""interface specification for objects that may behave like a basic tree.

Ignoring tree auxProperties / traits for now, not sure how best
to override them


TODO: breakpoints are probably still useful for parent/root encapsulation in
	nested tools; everything else should be done with wpdex and paths
	
	OH BOY now trying to work out breakpoints / virtual roots, everything I try feels
	wrong
	would it be deranged to check the tree's actual name? Like if it's called "root",
		treat it like a root?
	
TODO: GENERATOR branches
	and then ways to specify overrides on the generated data
	maybe we should just leave this to chimaera
"""

CHILD_LIST_T = VisitAdaptor.CHILD_LIST_T
PARAMS_T = VisitAdaptor.PARAMS_T
ChildData = VisitAdaptor.ChildData

keyT = Pathable.keyT
TreeType = T.TypeVar("TreeType", bound="TreeInterface")
class TreeInterface(Pathable,
                    Serialisable,
                    Visitable,
                    ):
	"""base class for tree-like objects -
	no requirements for internal storage or structure,
	only interface around name, value, branches and properties

	TODO: the different layers of setName(), _setRawName() etc are leftover from
	 trying to pack referencing and events into this single object.
	 they should probably be simplified
	"""

	def childObjects(self, params:PARAMS_T)->CHILD_LIST_T:
		"""maximum delegation to this tree's children"""
		return (
			ChildData("@N", self._getRawName() ),
			ChildData("@V", self._getRawValue() ),
			ChildData("@AUX", self._getRawAuxProperties() ),
			*(ChildData(i._getRawName(), i ) for i in self.branches)
		)

	@classmethod
	def newObj(cls, baseObj: TreeType, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		"""create a new tree object from a base object and child type list
		list of child branches should already have been regenerated
		"""
		#log("tree newObj", params, baseObj, type(baseObj))
		# log("tree childDatas", childDatas)
		#log(getattr(type(baseObj), "_generated", False))
		kwargs = {}
		if params.visitKwargs.get("serialParams", {}).get("PreserveUid"):
			if hasattr(baseObj, "uid"):
				kwargs["uid"] = baseObj.uid

		tree = type(baseObj)(
			name=childDatas[0][1], #name
			value=childDatas[1][1], # value
			**kwargs
		)

		tree._setRawAuxProperties(childDatas[2][1])
		for key, obj, data in childDatas[3:]:
			assert isinstance(obj, TreeInterface)
			tree.addBranch(obj)
		return tree



	class SerialKeys:
		name = "@NAME"
		value = "@VALUE"
		uid = "@UID"
		children = "@CHILDREN"
		address = "@ADDRESS"
		properties = "@PROPERTIES"
		type = "@TYPE"
		rootData = "@ROOT_DATA"
		format = "@FORMAT_VERSION"

		# layout constants
		layout = "@LAYOUT"
		nestedMode = 0
		flatMode = 1

	@classmethod
	def serialKeys(cls):

		return cls.SerialKeys

	class SignalKeys:
		NameChanged = "nameChanged"
		ValueChanged = "valueChanged"
		PropertyChanged = "propertyChanged"
		StructureChanged = "structureChanged"

	class AuxKeys(TypeNamespace):
		class _Base(TypeNamespace.base()):
			"""base aux keys for all trees"""
		class LookupCreate(_Base):
			"""if True, missing branches will be created when traversing"""
		class Default(_Base):
			"""default value for branches"""
		class Description(_Base):
			"""description of branch"""

	_DEFAULT_PROP_KEY = "_default"
	default = TreePropertyDescriptor( # don't need a sentinel here, trees can be None by default
		AuxKeys.Default, default=None, inherited=True)

	lookupCreate : bool = TreePropertyDescriptor(
		AuxKeys.LookupCreate, default=True, inherited=True)

	description : str = TreePropertyDescriptor(
		AuxKeys.Description, default="", inherited=False	)

	@classmethod
	def defaultBranchCls(cls):
		"""might be an idea to have an instance version of this,
		to define per-branch areas of the tree to inherit types,
		while other areas stay as defaults"""
		return cls

	# @classmethod
	# def defaultSignalCls(cls):
	# 	return TreeSignalComponent

	@classmethod
	def defaultAuxProperties(cls)->dict:
		return {}

	# def __init__(self):
	# 	"""now that tree inherits from pathable, we should switch around the obj / childObjects targeting"""
	# 	self.scratch = {}
	# 	if type(self) is TreeInterface: raise TypeError("Do not instantiate interface directly, use main Tree object instead")


	def __repr__(self):
		return "<{} ({}) : {}>".format(self.__class__, self.getName(), self.getValue())

	def __str__(self):
		return repr(self)
	def __eq__(self, other):
		""" equivalence considers value, branch names and extras
		for exact comparison use 'is' """
		if isinstance(other, TreeInterface):
			return self.isEquivalent(other)
		return False

	def __lt__(self, other):
		if not isinstance(other, TreeInterface):
			raise TypeError(
				"comparison not supported with {}".format(
					other, other.__class__))
		return self.name < other.name

	def isEquivalent(self, branch, includeBranches=False):
		""" tests if this tree is equivalent to
		the given object """
		flags = [
			self.name == branch.name,
			# self.value == branch.value,
			#self.extras == branch.extras
		]
		try:
			flags.append(self.value == branch.value)
		except (TypeError, NotImplementedError):
			return False
		if includeBranches:
			flags.append(all([i.isEquivalent(n, includeBranches=True)
				for i, n in zip(self.branches, branch.branches)]))
		return all(flags)

	#region name

	#TODO: I know all these layers are ridiculous, I'll simplify them once there's definitely no use for them
	def _getRawName(self)->str:
		"""OVERRIDE for backend"""
		return self._name
	def getName(self)->str:
		"""interface and outer function may be redundant, since getting tree's
		direct name should never trigger any complex computation"""
		return self._getRawName()

	def _setRawName(self, name:str):
		"""OVERRIDE for backend,
		if interface does not hold its own data"""
		raise NotImplementedError
	def validateNewName(self, name:str):
		"""OVERRIDE
		validate name before setting it"""
		return name
	def setName(self, name:str):
		"""OVERRIDE
		outer function to do any validation or processing
		before setting name value"""
		oldName = self.getName()
		if oldName == name: # no change, skip
			return
		# check name is valid
		name = self.validateNewName(name)
		# set name internally
		self._setRawName(name)

		return name

	@property
	def name(self)->str:
		return self.getName()
	@name.setter
	def name(self, val:str):
		self.setName(val)

	n = name # shorthand alias
	#endregion


	#region default value
	def _getDefault(self):
		"""return a static or callable object used in case of an empty
		value on this tree.
		On the normal Tree, this looks for the aux property "default"
		"""
		return self.getAuxProperty(self.AuxKeys.Default)
	def _evalDefault(self):
		if callable(self._getDefault()):
			return self._getDefault()(self)
		return self._getDefault()
	#endregion

	#region value
	def _getRawValue(self):
		""" OVERRIDE for backend, retrieving direct value held on this tree.
		Since None may be a valid value, return Sentinel.Empty
		if this tree has not had its value set."""
		return Sentinel.Empty
	def getValue(self):
		"""retrieve value stored in tree, or its default if empty"""
		raw = self._getRawValue()
		if raw is None and self._getDefault() is not None:
			try:
				self._setRawValue(self._evalDefault())
			except Exception as e:
				print("could not eval default for tree ", self, self._getDefault())
				raise e
		return self._getRawValue()

	def _setRawValue(self, value):
		""" OVERRIDE for backend
		inner function to set value without validation or processing.
		For now we emit signals from outer interface function,
		not sure if that's correct or we should emit them here,
		from the lowest level
		"""
		raise NotImplementedError
	def validateNewValue(self, value):
		"""
		validate new value before setting, or coerce to valid.
		:raises ValueError: if value is invalid

		TODO: consider using a validator object from wp, or allowing injection for it
		"""
		return value
	def setValue(self, value):
		"""outer function to do any validation or processing"""
		# oldValue = self.getValue()
		# if oldValue == value: # no change, skip
		# 	return
		# # check value is valid
		# value = self.validateNewValue(value)
		# # set value internally
		self._setRawValue(value)

	@property
	def value(self)->T:
		return self.getValue()
	@value.setter
	def value(self, val:T):
		self.setValue(val)
	v = value
	#endregion


	# region parent
	def _getRawParent(self)->TreeType:
		"""OVERRIDE for backend"""
		return self._parent
	def getParent(self)->TreeType:
		"""OVERRIDE for backend"""
		return self._getRawParent()

	# def _setParent(self, parentBranch:(TreeInterface, None)):
	# 	"""should be internal to addBranch(), not to be used in client code
	# 	set to None to remove branch from parent"""
	# 	raise NotImplementedError

	@property
	def parent(self)->TreeType:
		""":rtype AbstractTree"""
		return self.getParent()
	p = parent
	#endregion


	# region branches
	def _getRawBranches(self)->T.List[TreeType]:
		"""OVERRIDE for backend
		return live list of tree branches
		"""
		raise NotImplementedError

	def getBranches(self)->list[TreeType]:
		"""return a list of immediate branches of this tree"""
		return list(self._getRawBranches())
	def getBranchMap(self)->dict[str, TreeType]:
		"""return a dict of immediate branches of this tree, keyed by name"""
		return { b.getName() : b for b in self.getBranches()}
	@property
	def branches(self) -> list[TreeType]:
		"""return a list of immediate branches of this tree
		override here"""
		return self.getBranches()
	#endregion


	# region auxProperties
	# these were originally called just "properties" but this led to confusion
	def _setRawAuxProperties(self, props:dict):
		raise  NotImplementedError
	def _getRawAuxProperties(self)->dict:
		"""OVERRIDE for backend"""
		raise NotImplementedError
	@property
	def auxProperties(self) -> dict:
		"""return live dict, allowing direct setting of keys"""
		return self._getRawAuxProperties()

	def getAuxProperty(self, key: (str, AuxKeys), default=None):
		# if isinstance(key, self.AuxKeys):
		# 	key = str(key)
		key = str(key)
		return self.auxProperties.get(key, default)

	def setAuxProperty(self, key: str, value):
		self.auxProperties[key] = value

	def removeAuxProperty(self, key):
		if key in self.auxProperties:
			self.auxProperties.pop(key)
	#endregion


	#region traversal
	separatorChars = {
		"child" : "/",
		"attribute" : ".",
		#"parent" : "^",
	}
	parentChar = "^"


	def _branchesFromToken(self, token:keyT)->list[TreeType]:
		""" given single address token, return zero or more found direct branches
		TODO: add proper filtering, listing, wildcarding here etc
		 """
		if token == self.parentChar:
			return [self.parent]
		if found := self.branchMap().get(token):
			return [found]
		return []

	def getBranch(self, key:keyT,
	              traverseParams:defaultTraverseParamCls()=None)->(TreeType, None):
		"""return branch for this tree, or None"""
		try:
			#return self.traverse(key, traverseParams or self.defaultTraverseParamCls()())
			result = self.access(self, key, values=False)
			#log("getBranch result", result)
			return result
		except KeyError:
			return None

	def get(self, key:keyT, default=None, traverseParams:defaultTraverseParamCls()=None):
		"""return branch value, or default"""
		result = self.getBranch(key, traverseParams)
		if result:
			return result.value
		return default

	def _consumeFirstPathTokens(self, path: pathT, **kwargs) ->tuple[list[Pathable], pathT]:
		"""tree-specific syntax and wrapper for pathable logic -
		this might be better split up somehow"""
		if not path:
			return [self], path
		token, *path = path
		#if not token: #TODO: does wrong stuff if you pass it 0
		if isinstance(token, str):
			if not token: #TODO: does wrong stuff if you pass it 0
				return [self], path
		if token == ".." :
			return [self.parent], path
		# check if it's a special token
		#TODO: maybe formalise this somehow, but also maybe it's ok as jank
		if token.startswith("@"):
			if token == "@N" : return [self.name], path
			if token == "@V" : return [self.value], path
			if token == "@AUX" : return [self.auxProperties], path
			#return self.childObjects()

		# check if branch is directly found - return it if so
		found = self._branchesFromToken(token)
		if found:
			return found, path
		# no branch found

		# if branch should not be created, lookup is invalid
		create = kwargs.get("create", self.lookupCreate)
		#log("create", self.lookupCreate, create)
		if not create:
			raise KeyError("tree {} has no child {} in branches {}".format(self, token, self.branches))

		# create new child branch for lookup
		obj = self._createChildBranch(token)

		# add it to this tree
		self.addBranch(obj)
		return [obj], path

	def buildTraverseParamsFromRawKwargs(self, **kwargs) ->TraversableParams:
		"""build traverse params object from raw kwargs"""
		params : TreeTraversalParams = self.defaultTraverseParamCls()()
		create = kwargs.pop("create", None)
		if create is None:
			create = self.lookupCreate
		params.create = create
		return params

	def __call__(self, *path:keyT,
	             create=None,
	             one=None,
	             #traverseParams:defaultTraverseParamCls()=None,
	             **kwargs)->TreeInterface:
		""" index into tree hierarchy via address sequence,
		return matching branch"""
		#log("__call__", self, path, [type(i) for i in self.branches])

		try:
			result = self.access(self, path, values=False, create=create, **kwargs)
			#log("_call return", result)
		except Exception as e:
			print("unable to index path", path)
			raise e
		#log("__call__ END ", result, type(result))
		return result
	#endregion

	def __setitem__(self, key:(str, tuple), value:T, **kwargs):
		""" assuming that setting tree valueExpressions is far more frequent than
		setting actual tree objects """
		kwargs["create"] = True
		self(*self.toPath(key), **kwargs).value = value

	def __getitem__(self, address:(str, tuple),
	                )->T:
		""" returns direct value of lookup branch
		:rtype T
		"""
		return self(*self.toPath(address),
		            #address,
		            ).value


	# connected nodes
	def branchMap(self)-> dict[str, TreeType]:
		"""return a nice view of {tree name : tree}
		generated from uid map"""
		try:
			return {i.name : i for i in self.branches}
		except Exception as e:
			s = self.branches[0].name
			log("branchMap error", type(s), type(s).__hash__)
			raise e

	@property
	def uidBranchMap(self)->dict[str, TreeType]:
		"""return a nice view of {tree uid : tree}
		generated from uid map"""
		return {i.uid : i for i in self.branches}

	@property
	def isLeaf(self)->bool:
		return not self.branches

	@property
	def leaves(self)->list[TreeType]:
		"""returns branches under this branch
		which do not have branches of their own"""
		return [i for i in self.allBranches(False) if i.isLeaf]


	def keys(self)->tuple[str]:
		return tuple(self.branchMap().keys())

	@property
	def siblings(self)->list[TreeType]:
		if self.parent:
			l = self.parent.branches
			l.remove(self)
			return l
		return []

	def allBranches(self, includeSelf=True, depthFirst=True, topDown=True)->list[TreeType]:
		""" returns list of all tree objects
		depth first
		if topDown, reverse final list
		"""
		found = [ self ] if includeSelf else []
		if depthFirst:
			for i in self.branches:
				found.extend(i.allBranches(
					includeSelf=True, depthFirst=True, topDown=True))
		else:
			found.extend(self.branches)
			for i in self.branches:
				found.extend(i.allBranches(
					includeSelf=False, depthFirst=False, topDown=True))
		if not topDown:
			found = list(reversed(found))
		return found

	def _ownIndex(self)->int:
		if self.parent:
			return self.parent.index(self.name)
		else: return -1

	def index(self, lookup=None, *args, **kwargs)->int:
		if lookup is None: # get tree's own index
			return self._ownIndex()
		if lookup in self.branchMap().keys():
			return list(self.branchMap().keys()).index(lookup, *args, **kwargs)
		else:
			return -1

	def flattenedIndex(self)->int:
		""" return the index of this branch if entire tree were flattened """
		index = self.index()
		if self.parent:
			index += self.parent.flattenedIndex() + 1
		return index

	if T.TYPE_CHECKING:
		def trunk(self, includeSelf=True, includeRoot=True)->list[TreeType]: # no custom behaviour
			"""return sequence of ancestor trees in descending order to this tree"""
			#
			#
			# branches = []
			# current = self
			# while current.parent:
			# 	branches.insert(0, current)
			# 	current = current.parent
			# if includeRoot:
			# 	branches.insert(0, current)
			# if branches and not includeSelf:
			# 	branches.pop(-1)
			# return branches


	# def depth(self) -> int:
	# 	"""return int depth of this tree from root"""
	# 	return len(self.trunk(includeSelf=True, includeRoot=False))

	# # addresses
	# def address(self, includeSelf=True, includeRoot=False, uid=False)->list[str]:
	# 	"""if uid, return path by uids
	# 	else return nice string paths
	# 	recursive since different levels of tree might format their addresses
	# 	differently"""
	# 	trunk = self.trunk(includeSelf=includeSelf,
	# 	                   includeRoot=includeRoot,
	# 	                   )
	# 	if uid:
	# 		tokens = [i.uid for i in trunk]
	# 	else:
	# 		tokens = [i.name for i in trunk]
	# 	return tokens
	#
	#
	def stringAddress(self, includeSelf=True, includeRoot=False) -> str:
		""" returns the address sequence joined by the tree separator """
		trunk = self.trunk(includeSelf=includeSelf,
		                   includeRoot=includeRoot,
		                   )
		s = ""
		for i in range(len(trunk)):
			s += trunk[i].name
			if i != (len(trunk) - 1):
				s += trunk[i].separatorChars["child"]

		return s

	def commonParent(self, otherBranch: TreeType)->(TreeType, None):
		""" return the lowest common parent between given branches
		or None
		if one branch is direct parent of the other,
		that branch will be returned
		"""
		#log("commonParent", self.root, otherBranch.root, self.root is otherBranch.root)
		if self.root is not otherBranch.root:
			return None
		otherTrunk = set(otherBranch.trunk(includeSelf=True, includeRoot=True))
		# otherTrunk.add(otherBranch)
		test = self
		while test not in otherTrunk:
			test = test.parent
		return test

	def relAddress(self, fromBranch=None):
		""" retrieve the relative path from the given branch to this one"""
		fromBranch = fromBranch or self.root

		# check that branches share a common tree (root)
		#print("reladdress", self, self.trunk(includeSelf=True, includeRoot=True))
		common = self.commonParent(fromBranch)
		if not common:
			raise LookupError("Branches {} and {} "
			                  "do not share a common root".format(self, fromBranch))

		addr = []
		commonDepth = common.depth()
		# parent tokens to navigate up from other
		for i in range(commonDepth - fromBranch.depth()):
			addr.append(self.root.parentChar)
		# add address to this node
		addr.extend(
			self.address(includeSelf=True)[commonDepth:])
		return addr

	def __len__(self):
		return len(self.getBranches())

	def __bool__(self):
		"""prevent empty tree from evaluating as false"""
		return True

	def __contains__(self, item):
		"""check against exact branch object"""
		return isinstance(item, TreeInterface) and item in self.getBranches()

	def __iter__(self):
		"""iterate over branches"""
		return self.getBranches().__iter__()

	# region breakpoints
	"""keep embedded version very simple,
	more complexity can be handled in a tools lib.
	another approach is to use trees as values of other
	trees - not everything has to be contiguous
	"""
	def breakPoints(self):
		"""return a list of breakpoints for this tree"""
		return self.auxProperties.get("breakpoint", {})
	def getBreakPointBranch(self, key="main"):
		"""return breakpoint for given key"""
		test = self
		while test:
			if test.breakPoints().get(key):
				return test
			test = test.getParent()
		return None

	def setBreakPoint(self, key="main", val=True):
		"""set breakpoint for given key"""
		if not self.auxProperties.get("breakpoint"):
			self.auxProperties["breakpoint"] = {}
		self.auxProperties["breakpoint"][key] = val


	# endregion

	# region structure changes - parenting, removing

	"""TODO: consistent way of referring to / resolving adjacent branches, defaulting to self
	should always be parent modifying child"""

	def _setRawBranchIndex(self, branch:TreeType, index:int):
		"""reorder this tree's direct branch to index"""
		index = resolveSeqIndex(index, len(self.branches))
		self._getRawBranches().remove(branch)
		self._getRawBranches().insert(index, branch)

	def setIndex(self, index, branch=None):
		""" reorders tree branch to given index"""
		if self.parent is None:
			return
		if branch is None:
			return self.parent._setRawBranchIndex(self, index)
		return self._setRawBranchIndex(self.getBranch(branch), index)


	def _removeBranch(self, branch:TreeType):
		"""remove branch from this tree"""
		self._getRawBranches().remove(branch)
		branch._setParent(None)
		return branch

	def remove(self, address:(keyT, TreeInterface, None)=None,
	           ):
		"""removes address, or just removes the tree if no address is given"""
		#log("remove", address, self, self.parent)
		result = None
		if address:
			branch = self.getBranch(address)
			parent = branch.parent
			result = parent._removeBranch(branch)
		else:
			if self.parent is not None:
				parent = self.parent
				result = self.parent._removeBranch(self)
			else:
				raise ValueError("Cannot remove root branch")

		return result


	def _addBranch(self, newBranch:TreeInterface, index:int)->TreeType:
		"""OVERRIDE for backend"""
		self._getRawBranches().append(newBranch)
		newBranch._setParent(self)
		if index is not None:
			self._setRawBranchIndex(newBranch, index)
		#
		# self.sendEvent(TreeDeltas.Create(newBranch, self, newBranch.serialise()),
		#                key=self.SignalKeys.StructureChanged)
		# if self.getSignalComponent(create=False):
		# 	self.getSignalComponent(create=False).structureChanged.emit(
		# 		TreeDeltas.Create(newBranch, self,
		# 		                  newBranch.serialise())
		# 	)

		return newBranch

	def addBranch(self, newBranch:TreeInterface, index:int=None, force=False)->TreeType:
		"""called on parent to add new child node
		also calls _setParent() on new child
		RAISES ERROR if name already in branches - could allow it silently,
		but that raises ambiguity on new index, what happens to references etc

		this is a different signature to what we have in the base Pathable, but maybe
		it's actually fine for now
		"""

		if newBranch.uid in self.uidBranchMap:
			if force:
				newBranch.remove()
			else:
				raise KeyError(f"UID of new branch {newBranch} already in tree {self, self.address()}" + "\n" + f" keys {self.keys()}")

		if newBranch.name in self.keys():
			if force:
				self(newBranch.name).remove()
			else:
				raise KeyError(f"Name of new branch {newBranch} already in tree {self, self.address()}" + "\n" + f" keys {self.keys()}")

		# get correct index
		index = resolveSeqIndex(index if index is not None else len(self.branches), len(self.branches))
		self._addBranch(newBranch, index)
		return newBranch



	def _createChildBranch(self, name)->TreeType:
		"""called internally when a branch is created on lookup -
		for now don't support passing args for new branch,
		but probably wouldn't be tough"""
		#print("create child branch", name)
		obj = self.defaultBranchCls()(name=name)
		return obj

	#endregion

	# region other behaviour
	def getInherited(self, key,
	                 default=None,
	                 returnBranch=False,
	                 ):
		"""return first instance of key found in trunk auxProperties
		if returnBranch, returns the first node in trunk that
		includes key
		"""

		if key in self.auxProperties:
			if returnBranch:
				return self
			return self.auxProperties[key]
		if not self.getParent():
			return default
		return self.getParent().getInherited(key, default=default, returnBranch=returnBranch)

	#endregion

	#region serialisation
	def __deepcopy__(self, memo:dict)->TreeType:
		""":returns Tree"""
		return self.copy(copyUid=False)
	# 	return self.deserialise(copy.deepcopy(self.serialise()))

	def copy(self, copyUid=False, toType=None, usePickle=False, useDeepcopy=False)->TreeInterface:
		"""preserveUids is super dangerous as it leads to multiple elements having
		the same uid - leave this false unless you know what you're doing"""
		targetType = toType or type(self)
		if copyUid:
			# prevent any immediate damage being done to uid structure
			#with self.ignoreUidChangesInBlock():
			return targetType.deserialise(
				self.serialise(),
				serialParams={"preserveUid" : True})
				# if usePickle: return pickle.loads(pickle.dumps(self))
				# elif useDeepcopy: return copy.deepcopy(self, )
				# else: return self.deserialise(self.serialise(), preserveUid=True)
		return targetType.deserialise(self.serialise())

	# TODO: brother do you have enough serial functions yet
	#  brother do you want some more
	def _rootData(self)->dict:
		"""Returns any data needed to describe the whole tree
		in its serialised state -
		nothing related to format of information, that should all be kept
		in encoder classes"""
		return {
			#self.serialKeys().format : 0
		}

	def _baseSerialData(self, params:dict=None)->dict:
		"""return name, value, auxProperties for this branch only"""
		#log("_base serial data", params)
		serialKeys = self.serialKeys()
		data = { serialKeys.name : self.getName()
		         }
		#if self.value != self.default:
		data[serialKeys.value] = self._getRawValue()
		if self.auxProperties != self.defaultAuxProperties():
			data[serialKeys.properties] = self.auxProperties

		# check if type differs from parent - if so define it
		if self.parent and self.parent.__class__ != self.__class__:
			typeData = CodeRef.get(self.__class__)
			data[serialKeys.type] = typeData

		# sometimes we don't care about uid
		if (params or {}).get("PreserveUid", True):
			data[serialKeys.uid] = self.uid

		return data


	def _serialiseFlat(self, params:dict=None):
		"""return flat representation of tree, using address
		"""
		data = { tuple(self.address()) : self._baseSerialData(params)}
		#for i in self.branches:
		for i in self.allBranches(includeSelf=False):
			data[tuple(i.address())] = i._baseSerialData(params)
		return data

	def _serialiseNested(self, params:dict=None):
		"""inner function to call for main process"""
		data = self._baseSerialData(params)
		#print("serialiseNested", self, self.branches)
		if self.branches:
			data[self.serialKeys().children] = [i._serialiseNested(params) for i in self.branches]
		return data

	def _serialiseNestedOuter(self):
		"""outer function to call for main process"""
		data = self._serialiseNested()
		# add root data
		data[self.serialKeys().rootData] = self._rootData()
		data[self.serialKeys().layout] = self.serialKeys().nestedMode
		return data

	def _serialiseFlatOuter(self):
		data = self._serialiseFlat()
		# add root data
		data[self.serialKeys().rootData] = self._rootData()
		data[self.serialKeys().layout] = self.serialKeys().flatMode
		return data


	def serialiseSingle(self)->dict:
		"""ignores children, returns only serial data for this branch"""
		return self._baseSerialData()

	@classmethod
	def deserialiseSingle(cls, data:dict, preserveUid=False)->cls:
		"""returns only a single branch"""
		return cls._deserialiseFromData(data, preserveUid)

	@classmethod
	def _deserialiseFromData(cls, baseData:dict, preserveUid=False, loadType=True)->cls:
		"""given a block defining name, value, auxProperties, regenerate a single tree"""
		serialKeys = cls.serialKeys()
		# check if a specific tree subtype is needed
		if isinstance(loadType, type):
			treeCls = loadType
		# if loadType is True, load saved type
		elif loadType and serialKeys.type in baseData:
			# retrieve the reference to type, resolve it to get the actual type
			treeCls = CodeRef.resolve(baseData[serialKeys.type])
		else:
			treeCls = cls

		tree = treeCls(
			name=baseData[treeCls.serialKeys().name],
			value=baseData.get(treeCls.serialKeys().value),
			uid=baseData.get(treeCls.serialKeys().uid) if preserveUid else None
		)

		tree._properties = baseData.get(treeCls.serialKeys().properties, treeCls.defaultAuxProperties())

		return tree

	@classmethod
	def _deserialiseNested(cls, data:dict, preserveUid=False, preserveType=True)->cls:
		#print("DESERIALISE NESTED")
		baseTree = cls._deserialiseFromData(data, preserveUid=preserveUid, loadType=preserveType)
		# make sure to delegate properly to deserialised tree's type for rest of lookup
		branchCls = type(baseTree)

		# regen any other branches, add as children
		try:
			if branchCls.serialKeys().children in data:
				for i in data[branchCls.serialKeys().children]:
					#print("child deserialise", i)
					newChild = branchCls._deserialiseNested(i, preserveUid=preserveUid)
					#print("add new child", newChild, baseTree.getBranches())
					baseTree.addBranch(newChild)
					#raise
		except Exception as e:
			print("------------------")
			print("ERROR deserialising", baseTree)
			pprint.pprint(data)

			raise e

		return baseTree

	@classmethod
	def _deserialiseFlat(cls, data:dict, preserveUid=False, preserveType=True)->cls:
		branches = {
			address : cls._deserialiseFromData(
				serialData, preserveUid=preserveUid,
				loadType=preserveType)
			for address, serialData in data.items()
		}
		firstBranch = branches.pop(next(branches.keys()))
		for address, branch in branches.items():
			firstBranch(address[:-1]).addBranch(branch)
		return firstBranch


	uniqueAdapterName = "wpTreeInterface"

	@classmethod
	def defaultDecodeParams(cls) ->dict:
		return {
			"PreserveUid" : True,
			"PreserveType" : True,
		}


	def encode(self, encodeParams:dict=None)->dict:
		nested = True
		data = self._serialiseNested(encodeParams) if nested \
			else self._serialiseFlat(encodeParams)
		# add root data
		data[self.serialKeys().rootData] = self._rootData()
		data[self.serialKeys().layout] = self.serialKeys().nestedMode if nested else self.serialKeys().flatMode
		return data

	@classmethod
	def decode(cls, serialData: dict,
	                 decodeParams:dict, formatVersion=-1) -> TreeInterface:
		#print("tree interface decode")

		# todo: maybe have outer layer of params indexed by object type?
		preserveUid = decodeParams["PreserveUid"]
		preserveType = decodeParams["PreserveType"]
		rootData = serialData[cls.serialKeys().rootData]  # not used yet

		# cls._setCaching(False)
		layoutMode = serialData.get(cls.serialKeys().layout, cls.serialKeys().nestedMode)
		if layoutMode == cls.serialKeys().nestedMode:
			tree = cls._deserialiseNested(serialData, preserveUid=preserveUid, preserveType=preserveType)
		elif layoutMode == cls.serialKeys().flatMode:
			tree = cls._deserialiseFlat(serialData, preserveUid=preserveUid, preserveType=preserveType)
		else:
			raise ValueError(f"Unknown layout mode {layoutMode} for data {serialData}")
		# cls._setCaching(True)
		# tree.setCachedHierarchyDataDirty(True, True)
		return tree

	#endregion
	# region literal definitions

	@classmethod
	def fromNameValueBranchesProperties(cls,
	                                    name="",
	                                    value=None,
	                                    branches:list[TreeInterface]=None,
	                                    properties=None,)->cls:
		"""unified function to build trees from given elements"""
		branch = cls(name=name, value=value)
		for i in branches or []:
			branch.addBranch(i)
		for k, v in (properties or {}).items():
			branch.setAuxProperty(k, v)
		return branch

	@classmethod
	def _fromLiteralTuple(cls, literal:(tuple[str],
				tuple[str, T.Any],
				tuple[str, T.Any, list],
				tuple[str, T.Any, list, dict])
	                      ):
		"""create a tree from a literal definition"""
		assert len(literal)
		if isinstance(literal, str):
			literal = (literal,)
		tree = cls(name=literal[0])
		if len(literal) > 1:
			tree.setValue(literal[1])
		if len(literal) > 2:
			for i in literal[2]:
				tree.addBranch(cls.fromLiteral(i))
		if len(literal) > 3:
			for k, v in literal[3].items():
				tree.setAuxProperty(k, v)
		return tree

	@classmethod
	def _fromLiteralDict(cls, literal:dict, parent:TreeInterface=None):
		"""
		create only tree branches, no values specified until leaves

		{"root" : {
			"branchA" : {
				"leafA" : "value"
				},
			"branchB" : {}
			}

		:param literal:
		:return:
		"""





	@classmethod
	def fromLiteral(cls, literal:
		(
				tuple[str],
				tuple[str, T.Any],
				tuple[str, T.Any, list],
				tuple[str, T.Any, list, dict],
			dict[str, T.Any],
		)):
		"""create a tree from a literal definition
		may not use full serialisation - this would be user-defined
		as an argument or something

		:param literal: dict defining tree

		("root", 1, [  # name, value, branches
			("branch1", 2, []), # name, value, no branches
			("branch2", 3, [
				("leaf1", 4, [], {"prop1": 1}), # name, value, no branches, auxProperties
				("leaf2", p={"prop2": 2}), # arguments can be defined by kwargs
		
		for defining a structure, dict can also be used:
		
		{ "root" : {
			"branch" : {
				"leaf" : 1


		"""
		if isinstance(literal, (tuple, list, str)):
			return cls._fromLiteralTuple(literal)
		elif isinstance(literal, dict):
			return cls._fromLiteralDict(literal)
		# can't tell if this code is clean or revolting
		assert len(literal)
		if isinstance(literal, str):
			literal = (literal,)
		tree = cls(name=literal[0])
		if len(literal) > 1:
			tree.setValue(literal[1])
		if len(literal) > 2:
			for i in literal[2]:
				tree.addBranch(cls.fromLiteral(i))
		if len(literal) > 3:
			for k, v in literal[3].items():
				tree.setAuxProperty(k, v)
		return tree


	# endregion

	# region general utils
	def getUniqueBranchName(self, baseName):
		return incrementName(baseName, set(self.keys()) - {baseName})

	def displayStr(self, nested=True):
		seq = pprint.pformat( self.serialise(#nested=nested
		                                     ), sort_dicts=False
		                      )
		return seq

	def display(self, nested=True):
		print(self.displayStr(nested))
	# endregion




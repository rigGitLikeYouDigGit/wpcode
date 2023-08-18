from __future__ import annotations

import typing as T

from wptree import TreeInterface
from wplib.sequence import flatten
from wplib.object import UnHashableDict

#from setFnMap
from .cache import om
from . import bases, attr

if T.TYPE_CHECKING:
	from wpm import WN

# def getPlugInput(plug:om.MPlug)->om.MPlug:
# 	"""return the input plug of the given plug
# 	"""
# 	return plug.source()

class PlugMeta(type):
	"""Metaclass to initialise plug wrapper from mplug
	or from string"""
	objMap = UnHashableDict()

	def __call__(cls, plug:T.Union[str, om.MObject, WN])->WN:
		"""check if existing plug object exists
		"""

		if isinstance(plug, PlugTree):
			return plug
		# filter input to MPlug
		mPlug = None
		if isinstance(plug, om.MPlug):
			mPlug = plug

		elif isinstance(plug, str):
			# raw string of "nodeA.translate"
			assert "." in plug, "invalid plug name given: {} - must include a node name first".format(plug)
			#node = plug.split(".")[0]
			# tokens = ".".join(plug.split(".")[1:])
			mPlug = attr.getMPlug(plug)

		# no caching for now, found it to be more trouble than it's worth
		plug = super(PlugMeta, cls).__call__(mPlug)
		return plug

		# check for (node, plug) tie in register
		tie = (mPlug.node(), mPlug.name().split(".")[1:])

		if tie not in PlugMeta.objMap:
			plug = super(PlugMeta, cls).__call__(mPlug)
			#plug = PlugTree(mPlug)
			PlugMeta.objMap[tie] = plug
		return PlugMeta.objMap[tie]

		#
		# elif isinstance(plug, str): # passed "node.translate"
		# 	ass
		# else:
		# 	print("mobj", node, type(node))
		#
		# # check if MObject is known
		# if mobj in NodeMeta.objMap:
		# 	# return node object associated with this MObject
		# 	return NodeMeta.objMap[mobj]
		#
		# # get specialised WN subclass if it exists
		# postInitWrap = NodeMeta.wrapperClassForMObject(mobj)
		#
		# # create instance
		# ins = super(NodeMeta, postInitWrap).__call__(mobj)
		# # add to MObject register
		# NodeMeta.objMap[mobj] = ins
		# return ins

TreeType = T.TypeVar("TreeType", bound="PlugTree") # type of the current class

class PlugTree(
	#Tree,
	TreeInterface,
    bases.PlugBase,
			   ):
	"""specialised tree for interfacing with Maya plugs
	name is attribute name,
	value is lambda to look up plug value
	look up children lazily on demand

	attr lib operates on mplugs, defines logic for NtoN
	connections
	tree only implements it

	for now assume no data will ever be saved with this object,
	should be transient

	array plugs work like:
	array attr
	+- [1]
		+- translate
			+- translateX

	"""

	rootName = "plugRoot"

	trailingChar = "+" # signifies uncreated trailing plug index
	separatorChar =  "."

	def __init__(self,
				 plug:T.Optional[om.MPlug, str],
				 #_node:om.MObject=None,
				 #_nodeMFn:om.MFn=None,
				 ):
		""""""
		TreeInterface.__init__(self)

		self._MPlug = plug
		self.hType : attr.HType = attr.plugHType(self.MPlug)
		#self._parent : PlugTree = _parent # parent reference to reuse persistent tree objects

	@property
	def MPlug(self)->om.MPlug:
		return self._MPlug

	@property
	def node(self)->WN:
		"""WN is imported within function to avoid dependency loop
		should top node should also be plug root?
		No - it is more useful to have each top-level plug be its own root
		"""
		from wpm.core import WN
		return WN(self.MPlug.node())


	#region tree integration
	def getName(self) ->str:
		"""test for allowing plugs to have int names, not just strings"""
		result = attr.splitPlugLastNameToken(self._MPlug)
		if self.MPlug.isElement:
			return int(result)
		return result

	def getParent(self) ->PlugTree:
		parentPlug = attr.parentPlug(self.MPlug)
		if parentPlug is None:
			return None
		return PlugTree(parentPlug)


	def getBranches(self) ->list[PlugTree]:
		return [ PlugTree(v) for k, v in attr.subPlugMap(self.MPlug).items()]


	def __eq__(self, other:PlugTree):
		return self.MPlug is other.MPlug

	def __str__(self):
		return self.stringAddress()

	def __hash__(self):
		return hash((self.node, self.MPlug.name()))

	def stringAddress(self, includeRoot=True) -> str:
		"""reformat full path to this plug
		"""
		trunk = self.trunk(includeSelf=True,
		                   includeRoot=includeRoot,
		                   breakpoint=breakpoint)
		s = ""
		for i in range(len(trunk)):
			s += str(trunk[i].name)
			if i != (len(trunk) - 1):
				s += trunk[i].separatorChar

		return s

	def __repr__(self):
		return f"<PlugTree {str(self.MPlug.name())}>"

	@property
	def isLeaf(self)->bool:
		return not any((self.MPlug.isCompound, self.MPlug.isArray))


	def __call__(self, *address, _parsedAddress=None,
	             _addressExpands=None,
	             **kwargs)->(PlugTree, list[PlugTree]):
		""" parses lookup to return child plugs
		was just easier to override this entirely

		check first for array, then compound

		a compound array parent plug classes as both array and compound,
		but a compound array element plug is only compound

		_parsedAddress passed after tokens have been parsed the first time

		if any match or slice tokens are found, a list will be returned -
		else a single PlugTree

		"""
		if _parsedAddress is None:
			_parsedAddress = attr.parseEvalPlugTokens(attr.splitPlugTokens(*address))
			_addressExpands = attr.checkLookupExpands(address)

		#print("parsed address", _parsedAddress, "expands", _addressExpands)

		if not _parsedAddress:
			return self

		# if plug has no elements or children, this call is in error
		elif self.isLeaf:
			raise TypeError("No child plugs for simple plug", self.MPlug.name())

		currentToken = _parsedAddress.pop(0)
		plugs = attr.plugLookupSingleLevel(self.MPlug, token=currentToken)

		if plugs is None:
			raise KeyError("Incorrect token sequence {} for plug {}, check your calls".format(_parsedAddress, self.MPlug))

		# catch case where index is explicitly given for last array plug
		try:
			currentToken = int(currentToken)
			if currentToken == len(plugs):
				attr.newLastArrayElement(self.MPlug)
		except ValueError:
			pass

		# looking up or create corresponding branches
		resultBranches = []
		for plug in plugs:
			plugName = attr.splitPlugLastNameToken(plug)
			try:
				plugName = int(plugName)
			except ValueError:
				pass
			if not self.getBranch(plugName):
				newBranch = PlugTree(plug)
				#self.addChild(newBranch)
			try:
				resultBranches.append(self.branchMap[plugName])
			except KeyError:
				print( "key error", plugName, type(plugName), self.branchMap)
				raise
			#resultBranches.append(self.branchMap[plugName])

		# recurse if further tokens remain
		if _parsedAddress:
			# pass further calls into each found branch
			resultBranches = flatten(branch(_parsedAddress=_parsedAddress,
			                             _addressExpands=_addressExpands)
			                      for branch in resultBranches)
		return resultBranches if _addressExpands else resultBranches[0]

	# endregion

	def recache(self):
		"""dirty hack for testing"""
		try:
			PlugMeta.objMap.pop(((self.MPlug.node(), self.MPlug.name().split(".")[1:])))
		except:
			pass

	def getValue(self) ->T:
		"""retrieve MPlug value"""
		return attr.plugValue(self.MPlug)

	def setValue(self, val):
		"""if val is plugTree, connect plug
		if static value, set it"""
		if isinstance(val, PlugTree):
			attr.con(val.MPlug, self.MPlug)
		elif isinstance(val, om.MPlug):
			attr.con(val, self.MPlug)
		else:
			attr.setPlugValue(self.MPlug, val)

	# slightly more maya-familiar versions of the above
	def get(self):
		return self.getValue()

	def set(self, *value):
		if len(value) == 1:
			value = value[0]
		self.setValue(value)


	def nBranches(self):
		if self.MPlug.isArray:
			return self.MPlug.evaluateNumElements()
		if self.MPlug.isCompound:
			return self.MPlug.numChildren()
		return 0


	def arrayMPlugs(self):
		"""return MPlug objects for each existing element in index
		always returns 1 more than number of real plugs (as always
		one left open)
		"""

		if self.MPlug.evaluateNumElements() == 0:
			self.addNewElement()

		return [self.MPlug.elementByLogicalIndex(i)
				for i in range(self.MPlug.evaluateNumElements())]


	def element(self, index:int, logical=True):
		"""ensure that the given element index is present
		return that array plug"""
		return attr.arrayElement(self.MPlug, index)

	def addNewElement(self):
		"""extend array plug by 1"""
		attr.newLastArrayElement(self.MPlug)

	def ensureOneArrayElement(self):
		"""
		"""
		attr.ensureOneArrayElement(self.MPlug)



	plugParamType = (str, "PlugTree", om.MPlug)

	def _filterPlugParam(self, plug:plugParamType)->om.MPlug:
		"""return an MPlug from variety of acceptable sources"""
		plug = getattr(plug, "plug", plug)
		if isinstance(plug, PlugTree):
			plug = plug.MPlug
		elif isinstance(plug, str):
			plug = attr.getMPlug(plug)
		return plug


	def con(self, otherPlug:(plugParamType,
	                         list[plugParamType]),
	        _dgMod=None):
		"""connect this plug to the given plug or plugs"""
		dgMod = _dgMod or om.MDGModifier()

		# allow for passing multiple trees to connect to
		if not isinstance(otherPlug, (list, tuple)):
			otherPlug = (otherPlug,)

		for otherPlug in otherPlug:
			# check if other object has a .plug attribute to use
			otherPlug = self._filterPlugParam(otherPlug)
			attr.con(self.MPlug, otherPlug, dgMod)
		dgMod.doIt()

	# region networking
	def driver(self)->("PlugTree", None):
		"""looks at only this specific plug, no children
		returns PlugTree or None"""
		driverMPlug = self.MPlug.connectedTo(True,  # as Dst
		                       False  # as Src
		                       )
		if not driverMPlug: return None
		return PlugTree( driverMPlug[0])

	def drivers(self, includeBranches=True)->dict[PlugTree, PlugTree]:
		"""queries either immediate subplugs, or all subplugs to find set of all
		sub plugs
		returns { driverPlug : drivenPlug }

		"""
		checkBranches = self.allBranches(includeSelf=True) if includeBranches else [self]
		plugMap = {}

		for checkBranch in checkBranches:
			driver = checkBranch.driver()
			if driver:
				plugMap[driver] = checkBranch
		return plugMap

	def _singleDestinations(self)->tuple[PlugTree]:
		"""return all plugs fed by this single plug"""
		drivenMPlugs = self.MPlug.connectedTo(
			False, # as Dst
		    True # as Src
		)
		return drivenMPlugs

	def destinations(self, includeBranches=False)->dict[PlugTree, tuple[PlugTree]]:
		checkBranches = self.allBranches(includeSelf=True) if includeBranches else [self]
		plugMap = {}
		for checkBranch in checkBranches:
			plugMap[checkBranch] = checkBranch._singleDestinations()
		return plugMap

	def breakConnections(self, incoming=True, outgoing=True, includeBranches=True):
		"""disconnects all incoming / outgoing edges from this plug,
		or all of its branches"""
		#checkBranches = self.allBranches(includeSelf=True) if includeBranches else [self]
		dgMod = om.MDGModifier()
		if incoming:
			driverMap = self.drivers(includeBranches=includeBranches)
			for driver, driven in driverMap.items():
				dgMod.disconnect(driver.MPlug, driven.MPlug)
		if outgoing:
			driverMap = self.drivers(includeBranches=includeBranches)
			for driver, drivens in driverMap.items():
				for driven in drivens:
					dgMod.disconnect(driver.MPlug, driven.MPlug)
		dgMod.doIt()


	# endregion

	def driverOrGet(self)->PlugTree:
		"""if plug is driven by live input,
		return driving plug
		else return its static value"""
		if self.driver():
			return self.driver()
		return self.getValue()

	#def driveOrSet(self, plugOrValue):


	# region convenience
	@property
	def X(self):
		"""capital for more maya-like calls - node.translate.X"""
		return self.branches[0]

	@property
	def Y(self):
		return self.branches[1]

	@property
	def Z(self):
		return self.branches[2]

	@property
	def last(self):
		"""return the last available (empty) plug for this array"""
		return self(-1)

	#end


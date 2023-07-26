

"""file for objects indexed on creation in some way -
by name, uid etc
"""
from __future__ import annotations
import uuid

import typing as T
#from tree.lib.python import seedUid, bitwiseXor
from wplib.string import incrementName
#from tree.lib import uid as libuid


class IdElementBase:
	"""Base class for logic - reimplemented below"""

	keyT = (str, int)
	indexInstanceMap = {} # redefine in subclasses for separate maps

	def __init__(self, elementId=None):
		self._elementId = None
		self.setElementId(elementId or self.getNewElementId())

	def getElementId(self)->keyT:
		"""return the id of this element - name, uid, etc"""
		return self._elementId

	def _setElementIdInternal(self, newId):
		"""set the element id - no validation"""
		if self._elementId is not None:
			del self.getIndexInstanceMap()[self._elementId]
		self._elementId = newId
		self.getIndexInstanceMap()[newId] = self

	def setElementId(self, newId):
		"""do any outer validation needed here - then call private
		method to actually set it"""
		self._setElementIdInternal(newId)

	def getIndexInstanceMap(self)->dict[keyT, IdElementBase]:
		"""return the map of all instances of this class"""
		return self.indexInstanceMap

	@classmethod
	def getNewElementId(cls, instance:IdElementBase=None)->keyT:
		"""return a new element id"""
		raise NotImplementedError

	def getByIndex(self, index)->IdElementBase:
		"""return the element with the given index"""
		return self.getIndexInstanceMap().get(index)

	def _elementIndexNiceStr(self)->str:
		"""return a nicely formatted string for this element's index.
		in the uid case, you should probably curtail it"""
		return str(self.getElementId())

	def __hash__(self):  # remember to redeclare this in child classes if you also define __eq__
		return hash(self.getElementId())

	def __repr__(self):
		"""return this object by name"""
		return f"<{self.__class__.__name__}({self._elementIndexNiceStr()})>"

C = T.TypeVar("C", bound="UidElement")
class UidElement(IdElementBase):

	indexInstanceMap = {} # redefine in subclasses for separate maps

	def __init__(self, uid="", readable=True):
		super(UidElement, self).__init__(uid)

	@classmethod
	def getNewElementId(cls, instance:IdElementBase=None, seed=None, readable=False) ->keyT:
		if seed is not None:
			if readable:
				return libuid.getReadableUid(seed)
			return seedUid(seed)
		if readable:
			return  libuid.getReadableUid()
		return str(uuid.uuid4())

	# @classmethod
	# def byUid(cls:type[C], uid)->C:
	# 	return cls.idInstanceMap.get(uid)

	@property
	def uid(self)->str:
		return self.getElementId()

	@staticmethod
	def xorUids(a:str, b:str)->str:
		"""returns a new uid that is the bitwise xor of a and b
		this remains unique, but is consistent from inputs"""
		return bitwiseXor(a, b)

	def _elementIndexNiceStr(self) ->str:
		return self.uid[:4] + "-"

	# def setUid(self, uid, replace=True):
	# 	"""if replace, will override """
	# 	oldUid = self.uid
	# 	self.uid = uid
	# 	if self.idInstanceMap.get(oldUid):
	# 		self.idInstanceMap.pop(oldUid)
	# 	if self.idInstanceMap.get(uid):
	# 		if not replace:
	# 			return False
	# 	self.idInstanceMap[self.uid] = self
	# 	return True

class NamedElement(UidElement):
	"""same as above but with readable names"""
	nameInstanceMap = indexInstanceMap = {}

	defaultName = "newElementName"
	
	def __init__(self, name:str):
		super(NamedElement, self).__init__(name)

	@property
	def name(self):
		return self.getElementId()

	def setName(self, name):
		return super().setElementId(name)

	@classmethod
	def getNewElementId(cls, instance:IdElementBase=None, seed=None, readable=False) ->keyT:
		return incrementName(seed or cls.defaultName)






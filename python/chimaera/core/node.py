
from __future__ import annotations
import typing as T

from wplib import Expression, coderef
from wplib.sentinel import Sentinel
from wplib.object import UidElement

from wptree import Tree

if T.TYPE_CHECKING:
	from chimaera.core.graph import ChimaeraGraph

class ChimaeraNode(UidElement):
	"""smallest unit of computation in chimaera graph

	refmap is key : node filter, not too crazy
	"""

	def __init__(self):
		UidElement.__init__(self)
		self._parent : ChimaeraGraph = None

	@classmethod
	def typeRefStr(cls)->str:
		return coderef.getCodeRef(cls)

	@classmethod
	def defaultData(cls)->dict:
		attrMap : dict[str, Expression] = {
			"name" : Expression(name="name"),
			"value": Expression(name="value"),
		}
		refMapExp : T.Callable[[], dict[str, (str, ChimaeraNode)]] = Expression(
			value={},
			name="refMap")
		return {
			"attrMap" : attrMap,
			"refMap" : refMapExp,
		}

	@classmethod
	def setupNode(cls, node:ChimaeraNode, graph:ChimaeraGraph)->None:
		"""default process to set up node when freshly created -
		used by plug nodes to create plugs, etc
		"""


	def __str__(self):
		try:
			return f"<{self.__class__.__name__}({self.nameExp()})"
		except:
			return f"<{self.__class__.__name__}(UNABLE TO GET NAME - {self.getElementId()})>"


	def parent(self)->ChimaeraGraph:
		return self._parent

	def graph(self)->ChimaeraGraph:
		"""for now these are the same"""
		return self.parent()

	def dataBlock(self)->dict:
		return self.parent().nodeDataBlock(self)

	def nameExp(self)->Expression:
		return self.dataBlock()["attrMap"]["name"]
	def setName(self, name:str)->None:
		self.nameExp().setStructure(name)

	#region value and evaluation
	def valueExp(self)->Expression:
		return self.dataBlock()["attrMap"]["value"]
	def setValue(self, value)->None:
		self.valueExp().setStructure(value)

	def value(self):
		"""if exp is set, evaluate it
		if data tree is defined, return it,
		if params tree is defined, return it?

		Really like old version where a node either operated on
		data or became data - maybe we can carry that through
		"""
		if self.valueExp().rawStructure():
			return self.valueExp().resultStructure()

	# endregion

	#region refmap / node connections
	def refMapExp(self)->Expression:
		return self.dataBlock()["refMap"]

	def setRef(self, key, uid:str="", nodeFilter:str=""):
		"""updates expression source of refmap with given value"""
		refMap = self.refMapExp().rawStructure()
		if uid:
			valueStr = f"uid:{uid}"
		refMap[key] = valueStr
		self.refMapExp().setStructure(refMap)

	def refMap(self)->dict[str, (str, ChimaeraNode)]:
		"""returns resolved refmap dict of nodes"""
		return self.refMapExp().eval()

	def refMapRaw(self)->dict[str, str]:
		"""returns raw refmap dict of strings"""
		return self.refMapExp().rawStructure()

	def getRef(self, key, default=Sentinel.FailToFind, raw=False):
		"""return refmap entry for given key, or default if not found
		"""
		if raw:
			return self.refMapExp().rawStructure().get(key, default)
		return self.refMap().get(key, default)


	#endregion

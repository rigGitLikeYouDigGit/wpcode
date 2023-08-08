
from __future__ import annotations
import typing as T

from wplib import Expression
from wplib.object import UidElement

from wptree import Tree

if T.TYPE_CHECKING:
	from chimaera.core.graph import ChimaeraGraph

class ChimaeraNode(UidElement):
	"""smallest unit of computation in chimaera graph"""

	def __init__(self):
		UidElement.__init__(self)
		self._parent : ChimaeraGraph = None

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

	def __str__(self):
		try:
			return f"<{self.__class__.__name__}({self.nameExp()})"
		except:
			return f"<{self.__class__.__name__}(UNABLE TO GET NAME - {self.getElementId()})>"


	def parent(self)->ChimaeraGraph:
		return self._parent

	def dataBlock(self)->dict:
		return self.parent().nodeDataBlock(self)

	def nameExp(self)->Expression:
		return self.dataBlock()["attrMap"]["name"]
	def setName(self, name:str)->None:
		self.nameExp().setStructure(name)

	def valueExp(self)->Expression:
		return self.dataBlock()["attrMap"]["value"]
	def setValue(self, value)->None:
		self.valueExp().setStructure(value)



from __future__ import annotations
import typing as T

from wplib.sentinel import Sentinel
from wplib.object import DirtyNode, DirtyGraph
from wplib.expression.new import Expression

if T.TYPE_CHECKING:
	from wplib.expression.new import *

"""in more complex systems, allow expressions to track if they
need to be recomputed, if their inputs change etc"""

class DirtyExp( Expression, DirtyNode):

	def __init__(self,
	             value: [T.Callable, VT, str] = Sentinel.Empty,
	             policy: ExpPolicy = None,
	             name="exp",
	             ):
		DirtyNode.__init__(self)
		Expression.__init__(self, value=value, policy=policy, name=name)
		self._prevExpressions : tuple[(DirtyExp, DirtyNode)] = ()

	@classmethod
	def shortClsTag(cls) ->str:
		return "DExp"

	def _getDirtyNodeName(self) ->str:
		return self.getExpName()

	def getDirtyNodeAntecedents(self) ->tuple[DirtyNode]:
		"""get antecedents for dirty node"""
		return self._prevExpressions

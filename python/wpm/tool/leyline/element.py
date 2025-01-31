from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object import UidElement, DirtyNode, DirtyGraph
from wpm.tool.leyline.lib import DictModelled

if T.TYPE_CHECKING:
	from .graph import LLGraph


class LLElement(DictModelled, DirtyNode):
	"""base for LL elements - all having parents, all
	having antecedents"""

	def __init__(self,
	             name: str,  # take name as unique id for all ll elements for now
	             parents: dict[str, dict] = None,  # could be points, edges, faces etc
	             graph: LLGraph = None,
	             **kwargs
	             ):
		DirtyNode.__init__(self, name)
		DictModelled.__init__(
			self,
			name=name,
			parents=parents,
			**kwargs
		)
		self.graph = graph

	def __hash__(self):
		return id(self)

	def getDirtyNodeAntecedents(self) ->tuple[LLElement]:
		return tuple(self.graph.getEl(k) for k, v in self["parents"])

	def addParent(self, parent):
		self["parents"][self.graph.getId(parent)] = {}

	def removeParent(self, parent):
		self["parents"].pop(self.graph.getId(parent), None)

	def dirtyComputeFn(self):
		"""move around whenever node state is dirtied -
		ie whenever param changes in UI, OR when physically moves in scene?

		sync scene nodes to this object
		"""
		log("compute element", self)

	def onSceneChanged(self):
		"""call when element position in scene changes
		TODO: split this between moving with parent,
			and manual editing"""

	def onSceneEdited(self):
		"""call when changed by hand in scene?"""


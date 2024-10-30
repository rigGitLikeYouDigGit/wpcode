
from __future__ import annotations

import pprint
import typing as T


from PySide2 import QtCore, QtWidgets, QtGui

from chimaera import ChimaeraNode

from wpui.widget.canvas import *
from wpdex import *

from .node import NodeDelegate

if T.TYPE_CHECKING:
	from .view import ChimaeraView

class ChimaeraScene(WpCanvasScene):
	"""either ref() into graph data,
	or explicit function to add specific
	node -> get a delegate for it, add to scene, return
	delegate
	"""

	def __init__(self, graph:ChimaeraNode=None,
	             parent=None):
		super().__init__(parent=parent)

		self._graph : ChimaeraNode = rx(None)
		if graph:
			self.setGraph(graph)

	def graph(self)->ChimaeraNode:
		return self._graph.rx.value
	def rxGraph(self)->rx:
		return self._graph
	def setGraph(self, val:ChimaeraNode):
		self._graph.rx.value = val
		self.sync() # build out delegates

	def sync(self, elements=()):
		if not elements:
			self.clear()
		for name, node in self.graph().branchMap(): pass


	pass



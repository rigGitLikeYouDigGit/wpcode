
from __future__ import annotations

import pprint
import typing as T

from wplib import log, Sentinel, TypeNamespace
from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.uid import getUid4
from wplib.inheritance import clsSuper
from wplib.object import UidElement, ClassMagicMethodMixin, CacheObj
from wplib.serial import Serialisable

from PySide2 import QtCore, QtWidgets, QtGui

from chimaera import ChimaeraNode

from wpui.widget.canvas import *

from .node import NodeDelegate

if T.TYPE_CHECKING:
	from .view import ChimaeraView

class ChimaeraScene(WpCanvasScene):

	def __init__(self, graph:ChimaeraNode=None,
	             parent=None):
		super().__init__(parent=parent)

		self._graph : ChimaeraNode = None
		if graph:
			self.setGraph(graph)

	def graph(self)->ChimaeraNode:
		return self._graph
	def setGraph(self, val:ChimaeraNode):
		self._graph = val
		self.sync() # build out delegates

	def sync(self, elements=()):
		if not elements:
			self.clear()
		for name, node in self.graph().branchMap(): pass


	pass



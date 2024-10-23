
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

from wpui.widget.canvas import *

if T.TYPE_CHECKING:
	from .scene import ChimaeraScene

from .catalogue import NodeCatalogue

class ChimaeraView(WpCanvasView):

	if T.TYPE_CHECKING:
		def scene(self)->ChimaeraScene:pass

	def __init__(self, scene:ChimaeraScene, parent=None, ):
		super().__init__(parent=parent, scene=scene)
		self.catalogue = NodeCatalogue()

	def _onTabPressed(self):
		"""connect up to the tab hotkey, to create new nodes"""
		availNodes = self.scene().graph().availableNodeTypes()



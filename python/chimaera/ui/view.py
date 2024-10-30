
from __future__ import annotations

import pprint
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui
from wplib import log, Sentinel, TypeNamespace


from wpui.widget.canvas import WpCanvasView
from wpdex import *
from wpdex.ui import StringWidget
#from .catalogue import\ NodeCatalogue

if T.TYPE_CHECKING:
	from .scene import ChimaeraScene

class ChimaeraView(WpCanvasView):

	if T.TYPE_CHECKING:
		def scene(self)->ChimaeraScene:pass

	def __init__(self, scene:ChimaeraScene, parent=None, ):
		super().__init__(parent=parent, scene=scene)

		# simple node creation search bar - in future consider visual panes
		# might be useful to pull in different assets
		self.nodePaletteLine = StringWidget(
			value="",
			parent=self,
			options=self.scene().rxGraph().getAvailableNodeTypes().keys()
		)
		self.nodePaletteLine.setPlaceholderText("create node...")
		self.nodePaletteLine.hide()

		self.addKeyPressSlot(
			#self.KeySlot(lambda view : self.nodePaletteLine, keys=(QtCore.Qt.Key_Tab, ))
			self.KeySlot(self._onTabPressed, keys=(QtCore.Qt.Key_Tab, ))
		)
	def _onTabPressed(self, view):
		log("node options", self.scene().graph(), self.scene().graph().getAvailableNodeTypes())
		return self.nodePaletteLine
	def _onNodePaletteReturnPressed(self, s):
		if s in self.scene().graph().getAvailableNodeTypes():
			self.scene().graph().createNode(
				self.scene().graph().getAvailableNodeTypes()[s])


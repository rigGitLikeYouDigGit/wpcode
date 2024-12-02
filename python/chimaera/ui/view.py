
from __future__ import annotations

import traceback

from PySide2 import QtCore, QtGui, QtWidgets

from wpui.canvas import WpCanvasView
from wpdex import *
from wpdex.ui import StringWidget
#from .catalogue import\ NodeCatalogue
from wpui import lib as uilib

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
		self.nodePaletteLine.returnPressed.connect(self._onNodePaletteReturnPressed)


		self.addKeyPressSlot(
			#self.KeySlot(lambda view : self.nodePaletteLine, keys=(QtCore.Qt.Key_Tab, ))
			self.KeySlot(self._onTabPressed, keys=(uilib.SAFE_TAB_KEY, ))
		)

		self.testFlag = False
		self.scene().rxGraph().rx.watch(self._onGraphChanged,
		                                      onlychanged=False)


	def _onGraphChanged(self, *a):
		log("VIEW GRAPH CHANGED", a)

	def _onTabPressed(self, event:QtGui.QKeyEvent, view):
		# eventFilter in view catches first event, directly calls keyPressEvent, passes to checkFireKeySlots

		if self.nodePaletteLine.isVisible():
			self.nodePaletteLine.hide()
			return
		log("node options", self.scene().graph(),
		    self.scene().graph().getAvailableNodeTypes())
		#self.nodePaletteLine.setValue("")
		self.nodePaletteLine.setFocus()
		self.nodePaletteLine.selectAll()
		print("")
		log("tab pressed", self.nodePaletteLine.value(), self.nodePaletteLine._processValueForUi(self.nodePaletteLine.value()))

		return self.nodePaletteLine

	def _onNodePaletteReturnPressed(self):
		from .node import NodeDelegate
		s = self.nodePaletteLine.value()
		self.nodePaletteLine.hide()
		self.nodePaletteLine.clear()
		self.setFocus()
		if s not in self.scene().graph().getAvailableNodeTypes():
			log(f"unrecognised node type: {s}")
			return
		log("return pressed", s, self.scene().graph().getAvailableNodeTypes()[s])
		try:
			nodeType = self.scene().graph().getAvailableNodeTypes()[s]
			scene = self.scene()
			graph = scene.graph()
			log("scene", scene, "graph", graph)
			node = graph.createNode( nodeType
				)
		except Exception as e:
			log(f"error creating node of type {s}")
			traceback.print_exc()
			return
		# updating the scene data should automatically generate delegates
		# for new objects
		# move node to under cursor
		delegates = self.scene().delegatesForObj(node)
		# get the delegate for the node (assume it's the first one)
		log("DELEGATES", delegates)
		nodeDelegate = next(i for i in delegates if isinstance(i, NodeDelegate))
		log("delegate type", nodeDelegate, nodeDelegate.node)

		nodeDelegate.setPos(self.mapToScene(self._getMousePosForObjCreation()))
		# TODO: later implement Houdini behaviour if you already have a node selected -
		#   connect that node's main output to the new node's input



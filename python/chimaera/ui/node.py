

from __future__ import annotations

import pprint
import typing as T

from wplib import log, Sentinel, TypeNamespace
from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.uid import getUid4
from wplib.inheritance import clsSuper
from wplib.object import UidElement, ClassMagicMethodMixin, CacheObj, Adaptor
from wplib.serial import Serialisable

from PySide2 import QtCore, QtWidgets, QtGui

from wpui.widget.canvas import *

from chimaera import ChimaeraNode
from wpdex import WpDexProxy, WX

if T.TYPE_CHECKING:
	from .scene import ChimaeraScene
	from .view import ChimaeraView

refType = (WpDexProxy, WX)

class ReactLineEdit(QtWidgets.QLineEdit):
	"""test redoing the reactive stuff now we have an actual use case

	TODO: validation? schema?

	TODO: bring in the same structure as asset selector line here
		for tracking previous value, checking valid new one
	"""

	valueCommitted = QtCore.Signal(dict)

	def __init__(self, parent:QtWidgets.QWidget=None,
	             text:refType="",
	             #options=()
	             ):
		super().__init__(parent=parent)
		#log("LINE init")
		self.ref = text
		ref = text
		# if isinstance(ref, WpDexProxy):
		# 	ref = ref.ref(path=())
		self.setText(ref.rx.value)
		ref.rx.watch(self.setText)
		self.textEdited.connect(self._onTextEdited)
		self.editingFinished.connect(self._tryCommitText)
		self.valueCommitted.connect(lambda *args, **kwargs : ref.WRITE(*args, **kwargs))
		#self.valueCommitted.connect(self._onValueCommitted)

	def _onTextEdited(self, s:str):
		"""runs anytime user modifies text by hand at all -
		check validations here"""

	def _onValueCommitted(self, t):
		log("ON VALUE COMMITTED")

	def _tryCommitText(self,):
		"""can't muddy waters here - WRITE gets the exact new value,
		nothing more or less"""
		#log("TRY COMMIT TEXT")
		# oldValue = self.ref.rx.value
		# newValue = s
		self.valueCommitted.emit(self.text())


def paintDropdownSquare(rect:QtCore.QRect, painter:QtGui.QPainter,
          canExpand=True, expanded=False):
	painter.drawRect(rect)
	innerRect = QtCore.QRect(rect)
	innerRect.setSize(rect.size() / 3.0 )
	innerRect.moveCenter(rect.center())
	if expanded:
		rect.moveCenter(QtCore.QPoint(rect.center().x(), rect.bottom()))
	if canExpand:
		painter.drawRect(innerRect)




class TreePlugSpine(QtWidgets.QGraphicsItem):
	"""draw an expanding tree structure"""

	def __init__(self, parent=None, size=10.0):
		super().__init__(parent)
		self._expanded = False
		self.size = 10.0

	def childTreeSpines(self)->list[TreePlugSpine]:
		return [i for i in self.childItems() if isinstance(i, TreePlugSpine)]

	def expanded(self): return self._expanded
	def setExpanded(self, state): self._expanded = state; self.update()

	def boundingRect(self):
		return QtCore.QRectF(0, 0, self.size, self.size)
	def paint(self, painter:QtGui.QPainter, option, widget=...):
		"""draw a line backwards,
		a cross if not expanded,
		and the vertical bar down if expanded
		"""
		mid = self.size / 2.0
		painter.drawLine(-mid, mid, mid, mid)
		childSpines = self.childTreeSpines()
		if not self.childTreeSpines(): return
		paintDropdownSquare(rect=self.boundingRect(), painter=painter,
		                    canExpand=True, expanded=self.expanded())

expandPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Expanding,
	QtWidgets.QSizePolicy.Expanding
)
shrinkPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Maximum,
	QtWidgets.QSizePolicy.Maximum
)
class ShrinkWrapWidget(QtWidgets.QWidget):

	def sizeHint(self):
		baseRect = QtCore.QRect()
		for i in self.children():
			for at in ["boundingRect", "rect"]:
				try:
					r = getattr(i, at)()
					baseRect = baseRect.united(r)
					break
				except: continue
		size = baseRect.size()
		return size

class LabelWidget(QtWidgets.QWidget):
	"""simple way of showing a label alongside
	a normal widget"""

	def __init__(self, label="", w:QtWidgets.QWidget=None, parent=None):
		super().__init__(parent)
		self.w = w
		self.setLayout(QtWidgets.QHBoxLayout(self))
		self.label = QtWidgets.QLabel(label, parent=self)
		self.layout().addWidget(self.label)
		self.layout().addWidget(self.w)
		self.setContentsMargins(0, 0, 0, 0)
		self.label.setContentsMargins(0, 0, 0, 0)
		self.w.setContentsMargins(0, 0, 0, 0)

		self.setSizePolicy(expandPolicy)
		self.layout().setContentsMargins(0, 0, 0, 0)

class NodeDelegate(QtWidgets.QGraphicsItem, Adaptor):
	"""node has:
	- input tree of structures, widgets and plugs
	- central widgets for name, type, settings etc
	- output tree of structures, widgets and plugs

	adaptor allows defining new delegates for specific node types?

	sizing :)
	height:
		shrink central widgets as much as possible
	width :
		minimum of
			- central widgets
			- top port tree
			- bottom port tree
		dependency of width goes trees -> widgets

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (ChimaeraNode, )
	if T.TYPE_CHECKING:
		def scene(self)->ChimaeraScene: pass


	def __init__(self, node:ChimaeraNode,
	             parent=None,
	             ):
		super().__init__(parent)
		self.node = node

		# not making a whole special subclass to make this call less annoying
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)

		# central widget to contain views -
		# TODO: maybe make it easier to use multiple graphics widgets,
		#  map of proxy widgets and holders etc
		#  but this makes layout so much more difficult
		self.proxyW = QtWidgets.QGraphicsProxyWidget(parent=self
		                                             )
		self.w = ShrinkWrapWidget(parent=None)
		self.w.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
		self.w.setAutoFillBackground(False)
		self.setWidgetResult = self.proxyW.setWidget(self.w)
		self.wLayout = QtWidgets.QVBoxLayout(self.w)
		self.w.setLayout(self.wLayout)

		shrinkPolicy = QtWidgets.QSizePolicy(
			QtWidgets.QSizePolicy.Maximum,
			QtWidgets.QSizePolicy.Maximum
		)
		expandPolicy = QtWidgets.QSizePolicy(
			QtWidgets.QSizePolicy.Expanding,
			QtWidgets.QSizePolicy.Expanding
		)

		#self.w.setSizePolicy(shrinkPolicy)
		self.w.setSizePolicy(expandPolicy)

		self.nameLine = LabelWidget(
			label="name",
			w=ReactLineEdit(
				parent=self.w,
				#parent=self.w,
				text=self.node.ref("@N")),
            parent=self.w)


		self.wLayout.addWidget(self.nameLine)
		self.nameLine.setSizePolicy(shrinkPolicy)

		#self.syncSize()

	def syncSize(self):
		baseRect = self.proxyW.rect()
		expanded = baseRect.marginsAdded(QtCore.QMargins(10, 10, 10, 10))
		self.setRect(expanded)

	def boundingRect(self)->QtCore.QRectF:
		#return QtCore.QRectF(0, 0, 50, 50)
		baseRect = self.proxyW.rect()
		expanded = baseRect.marginsAdded(QtCore.QMargins(10, 10, 10, 10))
		return expanded

	def paint(self,
	          painter:QtGui.QPainter,
	          option:QtWidgets.QStyleOptionGraphicsItem,
	          widget=...):
		painter.drawRoundedRect(self.boundingRect(), 5, 5)
		brush = QtGui.QBrush(QtGui.QColor.fromRgbF(1, 1, 1))

		path = QtGui.QPainterPath()
		path.addRoundedRect(QtCore.QRectF(self.boundingRect()), 5, 5)
		painter.fillPath(path, brush)










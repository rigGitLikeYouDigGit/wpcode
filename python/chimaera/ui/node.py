

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
		# if isinstance(text, str):
		# 	self.setText(text)
		# else:
		self.ref = text
		ref = text
		# if isinstance(ref, WpDexProxy):
		# 	ref = ref.ref(path=())
		ref.rx.watch(self.setText)
		ref.rx.watch(fn=lambda *a, **kwargs : log("WATCHED", a, kwargs))
		#self.textEdited.connect(lambda *args, **kwargs : ref.WRITE(*args, **kwargs))
		self.textEdited.connect(self._onTextEdited)
		self.editingFinished.connect(self._tryCommitText)
		self.valueCommitted.connect(lambda *args, **kwargs : ref.WRITE(*args, **kwargs))

	def _onTextEdited(self, s:str):
		"""runs anytime user modifies text by hand at all -
		check validations here"""

	def _tryCommitText(self,):
		"""can't muddy waters here - WRITE gets the exact new value,
		nothing more or less"""
		# oldValue = self.ref.rx.value
		# newValue = s
		self.valueCommitted.emit(self.text())





class NodeDelegate(QtWidgets.QGraphicsItem, Adaptor):
	"""node has:
	- input tree of structures, widgets and plugs
	- central widgets for name, type, settings etc
	- output tree of structures, widgets and plugs

	adaptor allows defining new delegates for specific node types?

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

		# central widget to contain views -
		# TODO: maybe make it easier to use multiple graphics widgets,
		#  map of proxy widgets and holders etc
		#  but this makes layout so much more difficult
		self.proxyW = QtWidgets.QGraphicsProxyWidget(parent=self)
		self.w = QtWidgets.QWidget(parent=None)
		self.proxyW.setWidget(self.w)
		self.wLayout = QtWidgets.QVBoxLayout()
		self.w.setLayout(self.wLayout)

		self.nameLine = ReactLineEdit(
			#parent=self.w,
			parent=None,

		                              text=self.node.ref("@N"))
		#self.wLayout.addWidget(self.nameLine)
		#
		# self.syncSize()

	def syncSize(self):
		baseRect = self.proxyW.rect()
		expanded = baseRect.marginsAdded(QtCore.QMargins(10, 10, 10, 10))
		self.setRect(expanded)

	def boundingRect(self)->QtCore.QRectF:
		return QtCore.QRectF(0, 0, 50, 50)
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










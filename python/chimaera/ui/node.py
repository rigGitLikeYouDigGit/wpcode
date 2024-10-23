

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

refType = (WpDexProxy, WX)

class ReactLineEdit(QtWidgets.QLineEdit):
	"""test redoing the reactive stuff now we have an actual use case

	TODO: validation? schema?
	"""

	def __init__(self, parent:QtWidgets.QWidget=None,
	             text:refType="",
	             #options=()
	             ):
		super().__init__(parent=parent)
		ref = text
		if isinstance(ref, WpDexProxy):
			ref = ref.ref(path=())
		ref.rx.watch(self.setText)
		self.textEdited.connect(ref.WRITE)




class NodeDelegate(QtWidgets.QGraphicsRectItem, Adaptor):
	"""node has:
	- input tree of structures, widgets and plugs
	- central widgets for name, type, settings etc
	- output tree of structures, widgets and plugs

	adaptor allows defining new delegates for specific node types?

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (ChimaeraNode, )
	def __init__(self, node:ChimaeraNode,
	             parent=None,
	             ):
		super().__init__(parent)
		self.node = node

		# central widget to contain views -
		# TODO: maybe make it easier to use multiple graphics widgets,
		#  map of proxy widgets and holders etc
		#  but this makes layout so much more difficult
		self.proxyW = QtWidgets.QGraphicsProxyWidget(parent=self)
		self.w = QtWidgets.QWidget(parent=None)
		self.proxyW.setWidget(self.w)
		self.wLayout = QtWidgets.QVBoxLayout()
		self.w.setLayout(self.wLayout)

		self.nameLine = ReactLineEdit(parent=self.w,
		                              text=self.node.ref("@N"))
		self.wLayout.addWidget(self.nameLine)

		self.syncSize()

	def syncSize(self):
		baseRect = self.proxyW.rect()
		expanded = baseRect.marginsAdded(QtCore.QMargins(10, 10, 10, 10))
		self.setRect(expanded)










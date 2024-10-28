from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui

from wpdex import *


class TabWidget(QtWidgets.QTabWidget):
	"""unsure if this should be a proper reactive/ atomic widget,
	for now just specify a function when a new tab is created

	some complexity from allowing use with a backend model item, or
		just in pure widgets


	"""

	def __init__(self, parent=None,
	             value:T.Sequence[QtWidgets, T.Any]=None,
	             widgetForItemFn:T.Callable[[T.Any, "TabWidget", (str, None)], QtWidgets.QWidget]=None,
	             nameForItemFn:T.Callable[[(T.Any, QtWidgets.QWidget), "TabWidget"], str]=None,
	             requestNameOnNewTab=True,
	             newItemRequestedFn:T.Callable[["TabWidget", (str, None)], T.Any]=None
	             #newItemFn:T.Callable[[TabWidget, (str, None)], QtWidgets.QWidget]=None
	             ):
		QtWidgets.QTabWidget.__init__(self, parent)
		self._value = rx(value)
		self.requestNameOnNewTab = requestNameOnNewTab

	def addItem(self, ): pass


	def widgetForItemTemplateFn(self,
	                      item:T.Any,
	                      w:TabWidget,
	                      name:(str, None))->QtWidgets.QWidget:
		"""return a new widget for the given model item - """

	def nameForItemTemplateFn(self,
	                          item:(T.Any, QtWidgets.QWidget)):pass



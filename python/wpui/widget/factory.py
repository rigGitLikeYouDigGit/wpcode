from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

"""
widgets that dynamically get their own children
based on generation rules - 
TODO: consider if there's a way to do this fully procedurally,
maybe by radically extending the wpdex layouts

yeah I think that's the way, still doing this as a specific temp application
to get idem working
"""


class GenGroupBox(QtWidgets.QGroupBox):
	def __init__(self, title="", parent=None,
	             vertical=True,
	             getWidgetsFn:T.Callable[[GenGroupBox], list[QtWidgets.QWidget]]=None,
	             integrateFn:T.Callable[[GenGroupBox, QtWidgets.QWidget, int], None]=None,
	             deleteWidgetsOnSync=True,

	             ):
		super().__init__(title, parent)

		self.l = QtWidgets.QVBoxLayout(self) if vertical else QtWidgets.QHBoxLayout(self)
		self.deleteWidgetsOnSync = deleteWidgetsOnSync
		self.getWidgetsFn = getWidgetsFn # returns all the widgets to add
		self.integrateFn = integrateFn # runs over each widget once added to layout
		self.setLayout(self.l)
		self.setContentsMargins(1, 1, 1, 1)

	def addAction(self, title:str, fn=lambda s:s):
		"""hook up a callable function for when this action is clicked"""
		if isinstance(title, QtWidgets.QAction):
			return super().addAction(title)
		action = QtWidgets.QAction(title, self)
		l = lambda *a : fn(title) # annoying to make sure the ref never goes out of scope for Qt
		action.fn = l
		action.triggered.connect(l)
		return super().addAction(action)

	def sync(self):
		#log("gb sync")
		# clear existing widgets
		for i in range(self.l.count()):
			it = self.l.takeAt(i)
			if not it:
				continue
			w = it.widget()
			w.setParent(None)
			if self.deleteWidgetsOnSync:
				w.deleteLater()
			#w.hide()

		# generate widgets to add
		if not self.getWidgetsFn:
			return
		for i in self.getWidgetsFn(self):
			#log("gb sync", i)
			if isinstance(i, QtWidgets.QWidget):
				i.show()
				#log("after show")
				self.l.addWidget(i)
				#log("after addWidget")
			elif isinstance(i, QtWidgets.QAction):
				self.addAction(i)
			elif isinstance(i, str):
				self.addAction(i)
			elif isinstance(i, tuple):
				self.addAction(*i)
		self.layout().update()
		self.update()
		#log("after gb layout")
		# integrate widgets
		if not self.integrateFn:
			return
		for i in range(self.l.count()):
			w = self.l.itemAt(i).widget()
			self.integrateFn(self, w, i)

	def widgets(self):
		return [self.l.itemAt(i).widget()
		        for i in range(self.l.count())]



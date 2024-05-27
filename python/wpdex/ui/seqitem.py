
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log

from wplib.constant import SEQ_TYPES

from wpui import view as libview

from wpdex import WpDex, SeqDex
from wpdex.ui.base import WpDexWidget, WpTypeLabel


class DragItemWidget(QtWidgets.QWidget):
	"""small holder widget to allow dragging model items
	up and down"""
	def __init__(self, innerWidget:QtWidgets.QWidget, parent=None):
		super().__init__(parent)
		self.dragHandle = QtWidgets.QLabel("☰", parent=self)
		self.innerWidget = innerWidget
		#self.innerWidget.setParent(self)
		#layout = QtWidgets.QHBoxLayout()
		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self.dragHandle)
		layout.addWidget(innerWidget)
		self.setLayout(layout)
		self.setAutoFillBackground(True)

class SeqDexWidget(WpDexWidget):
	"""view for a list

	drag behaviour -
	drop directly on top of an item: override or swap?
	drop between items: insert

	or...
	ignore drag/drop for now, emulate something like a cursor in vim

	click an item to select it
	press enter to edit it
	ctl-v overwrites a selected item
	press alt to switch to cursor between items?

	work with selection spans / cursor spans - each has a start and end index


	"""
	forTypes = (SeqDex,)
	# init and build methods
	def buildChildWidgets(self):
		"""populate childWidgets map with widgets
		for all dex children"""
		dex = self.dex()
		#log("buildChildWidgets")
		for key, child in self.dex().children.items():
			childWidgetType = self.childWidgetType(child)

			childWidget = childWidgetType(child,
			                              parent=self
			                              )
			log("childWidget", childWidget, vars=0)
			self.childWidgets[key] = childWidget
	def buildExtraWidgets(self):
		"""build and populate model and view"""
		#super().buildExtraWidgets()
		self.typeLabel = WpTypeLabel(parent=self, closedText="[...]",
		                             openText="[")
		self.model = QtGui.QStandardItemModel(parent=self)
		#self.view = QtWidgets.QListView(parent=self)
		self.view = QtWidgets.QTreeView(parent=self)
		self.view.setModel(self.model)

		for i, (key, w) in enumerate(self.childWidgets.items()):
			# create an item, set its indexwidget
			self.model.appendRow(QtGui.QStandardItem(str(key)))
		for i, (key, w) in enumerate(self.childWidgets.items()):

			#dragWidget = DragItemWidget(w, parent=self)
			#dragWidget = DragItemWidget(QtWidgets.QLabel("TEST"), parent=self)
			#dragWidget = QtWidgets.QLabel("TEST")
			#dragWidget = QtWidgets.QLabel("TEST", parent=self) # works
			dragWidget = w
			dragWidget.setAutoFillBackground(True)

			self.view.setIndexWidget(self.model.index(i, 0), dragWidget)
			# label = QtWidgets.QLabel(str(key), parent=None)
			# self.view.setIndexWidget(self.model.index(i, 0), label)

		self.view.setSizeAdjustPolicy(
			QtWidgets.QAbstractItemView.AdjustToContents
		)
		#self.view.sizeHintForRow(0)

		libview.syncViewLayout(self.view)


	def buildLayout(self):
		layout = QtWidgets.QHBoxLayout()
		#layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self.typeLabel)
		layout.addWidget(self.view)
		self.setLayout(layout)


	# def sizeAdjustPolicy(self):
	# 	"""return the size adjust policy"""
	# 	print("size adjust policy")
	# 	return QtWidgets.QAbstractItemView.AdjustToContents


# class ListWpItemWidget(WpItemWidget):
# 	"""widget for a list"""
# 	forTypes = SEQ_TYPES
#
# 	superView : ListWpDexView
#
# 	def __init__(self, superItem:ListSuperItem, parent=None,
# 	             expanded=True):
# 		super().__init__( superItem, parent=parent)
#
# 		# initialise view
# 		self.superViewType = superItem._getComponentTypeForObject(
# 			self.superItem.wpPyObj, "view")
# 		self.superView = self.superViewType(
# 			self.superItem, parent=self
# 		)
# 		self.superView.setModel(self.superItem.wpChildModel)
# 		self.superView.buildIndexWidgets()
# 		# layout = QtWidgets.QVBoxLayout(self)
# 		# layout.addWidget(self.superView)
# 		# self.setLayout(layout)
#
# 		self.makeLayout()
#
# 		log("list w init", self.typeLabel)
#
# 		# set expanded state
# 		#self.setExpanded(expanded)
#
# 		#self.resize(self.sizeHint())
#
#
# 		#self.resize(300, 300)
# 		#self.setFixedSize(300, 300)
# 		#self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
#
#
#
#
# 		# # make expanded widget
# 		# self.expandedWidget = QtWidgets.QWidget(parent=self)
# 		# # bookend
# 		# self.firstBookendLabel = QtWidgets.QPushButton(
# 		# 	self.superItem.getBookendChars()[0],
# 		# 	parent=self.expandedWidget)
# 		# self.firstBookendLabel.clicked.connect(self.toggleExpanded)
# 		#
# 		# # initialise view
# 		# self.superViewType = superItem._getComponentTypeForObject(
# 		# 	self.superItem.wpPyObj, "view")
# 		# self.superView = self.superViewType(
# 		# 	self.superItem, parent=self.expandedWidget
# 		# )
# 		# #log("list widget init", self.superView, self.superItem.wpPyObj[0])
# 		# self.superView.buildIndexWidgets()
# 		# self.superView.resize(300, 300)
# 		# self.expandedWidget.resize(200, 200)
# 		#
# 		#
# 		# expandedLayout = QtWidgets.QHBoxLayout(self.expandedWidget)
# 		# expandedLayout.addWidget(self.firstBookendLabel)
# 		# expandedLayout.addWidget(self.superView)
# 		# expandedLayout.setAlignment(self.firstBookendLabel, QtCore.Qt.AlignTop)
# 		# self.expandedWidget.setLayout(expandedLayout)
# 		#
# 		# # make collapsed widget
# 		# # closed bookends (replace with typing summary)
# 		# self.stowedBookendLabel = QtWidgets.QPushButton(
# 		# 	self.superItem.getBookendChars()[0] + " ... " + self.superItem.getBookendChars()[1],
# 		# 	parent=self)
# 		# self.stowedBookendLabel.clicked.connect(self.toggleExpanded)
# 		#
# 		# # make stacked layout
# 		# # self.layout = QtWidgets.QStackedLayout(self)
# 		# # self.layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
# 		# # self.layout.addWidget(self.stowedBookendLabel)
# 		# # self.layout.addWidget(self.expandedWidget)
# 		# # self.setLayout(self.layout)
# 		# layout = QtWidgets.QVBoxLayout(self)
# 		# layout.addWidget(self.stowedBookendLabel)
# 		# layout.addWidget(self.expandedWidget)
# 		# self.setLayout(layout)
# 		#
# 		# layout.setAlignment(self.stowedBookendLabel, QtCore.Qt.AlignTop)
# 		# # layout.setStretch(0, 1)
# 		# # layout.setStretch(1, 1)
# 		#
# 		# # set model from item
# 		# self.superView.setModel(self.superItem.wpItemModel)
# 		#
# 		# # appearance
# 		# self.setAutoFillBackground(True)
#
#
#
# 	def isExpanded(self) -> bool:
# 		"""return True if expanded"""
# 		return self.expandedWidget.isVisible()
#
# 	def setExpanded(self, expanded:bool):
# 		"""set expanded state"""
# 		return
# 		self.expandedWidget.setVisible(expanded)
# 		self.stowedBookendLabel.setVisible(not expanded)
# 		#self.updateGeometry()
# 		#self.setGeometry(self.expandedWidget.geometry() if expanded else self.stowedBookendLabel.geometry())
# 		self.superView.scheduleDelayedItemsLayout()
# 		print("set expanded", expanded)
# 		self.superView.executeDelayedItemsLayout()
#
#
# 	def toggleExpanded(self, *args, **kwargs):
# 		"""toggle expanded state"""
# 		self.setExpanded(not self.isExpanded())




from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wplib import log

from wplib.constant import LITERAL_TYPES, SEQ_TYPES
from wpui.superitem import SuperModel, SuperItem, SuperItemView, SuperItemWidget


class ListSuperModel(SuperModel):
	"""model for a list"""
	forTypes = SEQ_TYPES


class ListSuperItem(SuperItem):
	"""superitem for a list"""
	forTypes = SEQ_TYPES

	wpPyObj : list

	# @classmethod
	# def getBookendChars(cls, forInstance:SuperItem=None) ->tuple[str, str]:

	def getBookendChars(self, forInstance: SuperItem = None) -> tuple[str, str]:
		"""return the characters to use as bookends for this item -
		"[", "]" for lists, "{", "}" for dicts, etc
		"""
		if isinstance(self.wpPyObj, list):
			return ("[", "]")
		if isinstance(self.wpPyObj, tuple):
			return ("(", ")")
		if isinstance(self.wpPyObj, set):
			return ("{", "}")
		raise ValueError(f"unsupported type for bookends{type(self.wpPyObj)}")

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		return self.wpVisitAdaptor.newObj(
			self.wpPyObj,
			((i[0].wpResultObj(), i[1]) for i in self.wpChildSuperItems())
		)

class ListSuperItemView(SuperItemView):
	"""view for a list"""
	forTypes = SEQ_TYPES

	def __init__(self, superItem:ListSuperItem, parent=None):
		super().__init__(superItem, parent=parent)
		#log(f"ListSuperItemView.__init__({superItem}, {parent})")

	def buildIndexWidgets(self):
		"""build all widgets for child items"""
		#log("ListSuperItemView.buildIndexWidgets", self.superItem.wpChildSuperItems())
		for i, (childItem, childType) in enumerate(self.superItem.wpChildSuperItems()):
			childItem : SuperItem
			childWidgetType = self.superItem._getComponentTypeForObject(
				childItem.wpPyObj, "widget")
			childWidget = childWidgetType(childItem, parent=self)
			#self.indexWidgets.append((childWidget, childType))


			# set as index widget
			self.setIndexWidget(childItem.index(), childWidget)

			# background colour
			childWidget.setAutoFillBackground(True)



class ListSuperItemWidget(SuperItemWidget):
	"""widget for a list"""
	forTypes = SEQ_TYPES

	superView : ListSuperItemView

	def __init__(self, superItem:ListSuperItem, parent=None,
	             expanded=True):
		super().__init__( superItem, parent=parent)

		# make expanded widget
		self.expandedWidget = QtWidgets.QWidget(parent=self)
		# bookend
		self.firstBookendLabel = QtWidgets.QPushButton(
			self.superItem.getBookendChars()[0],
			parent=self.expandedWidget)
		self.firstBookendLabel.clicked.connect(self.toggleExpanded)

		# initialise view
		self.superViewType = superItem._getComponentTypeForObject(
			self.superItem.wpPyObj, "view")
		self.superView = self.superViewType(
			self.superItem, parent=self.expandedWidget)
		#log("list widget init", self.superView, self.superItem.wpPyObj[0])
		self.superView.buildIndexWidgets()
		self.superView.resize(100, 100)
		self.expandedWidget.resize(200, 200)


		expandedLayout = QtWidgets.QHBoxLayout(self.expandedWidget)
		expandedLayout.addWidget(self.firstBookendLabel)
		expandedLayout.addWidget(self.superView)
		self.expandedWidget.setLayout(expandedLayout)

		# make collapsed widget
		# closed bookends (replace with typing summary)
		self.stowedBookendLabel = QtWidgets.QPushButton(
			self.superItem.getBookendChars()[0] + " ... " + self.superItem.getBookendChars()[1],
			parent=self)
		self.stowedBookendLabel.clicked.connect(self.toggleExpanded)

		# make stacked layout
		# self.layout = QtWidgets.QStackedLayout(self)
		# self.layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
		# self.layout.addWidget(self.stowedBookendLabel)
		# self.layout.addWidget(self.expandedWidget)
		# self.setLayout(self.layout)
		layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.stowedBookendLabel)
		layout.addWidget(self.expandedWidget)
		self.setLayout(layout)

		# set model from item
		self.superView.setModel(self.superItem.wpItemModel)


		# set expanded state
		self.setExpanded(expanded)


	def isExpanded(self) -> bool:
		"""return True if expanded"""
		return self.expandedWidget.isVisible()

	def setExpanded(self, expanded:bool):
		"""set expanded state"""
		self.expandedWidget.setVisible(expanded)
		self.stowedBookendLabel.setVisible(not expanded)
		self.superView.scheduleDelayedItemsLayout()
		self.superView.executeDelayedItemsLayout()


	def toggleExpanded(self, *args, **kwargs):
		"""toggle expanded state"""
		self.setExpanded(not self.isExpanded())



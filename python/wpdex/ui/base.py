


from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from dataclasses import dataclass

#import pretty_errors

from wplib import log
from wplib.object import Adaptor, VisitAdaptor
from wplib.serial import SerialAdaptor, Serialisable, serialise, deserialise

from wpdex import WpDex
from wpui.model import iterAllItems
#from wpui.typelabel import TypeLabel

"""

visualisation system for wpDex data structures - 
use paths to control params and overrides for widgets

new wpdex spine lets us simplify qt side - widgets are
effective views of the wpdex model, and need less
hardcoding in structure.

Use path overrides on main wpdex or widgets to control
different widget types shown for the same data, if needed

STORE NO STATE IN UI

"""

if T.TYPE_CHECKING:
	pass

class WpView:
	"""useful base class for a wpdex view -
	syncItemLayout() took a LONG time to work out
	move to proper placement somehow
	"""
	def syncItemLayout(self):
		"""sync the layout of this view with the item model"""
		self.scheduleDelayedItemsLayout()
		self.executeDelayedItemsLayout()

class WpTypeLabel(QtWidgets.QLabel):
	"""label for a wpdex item type, shows debug information on tooltip,
	acts to open and collapse item on click"""

	def __init__(self, parent:WpDexWidget=None,
	             closedText="[...]", openText="[",
	             openState=True):
		super(WpTypeLabel, self).__init__(parent=parent)
		self.closedText = closedText
		self.openText = openText
		self.setToolTip(
			f"{type(self.parent().dex())}\n"
			f"{self.parent().dex().path}")
		self.setOpenState(openState)

	if T.TYPE_CHECKING:
		def parent(self)->WpDexWidget:
			"""return parent widget"""
			return self.parent()

	def setOpenState(self, state:bool):
		"""set open state of label"""
		if state:
			self.setText(self.openText)
		else:
			self.setText(self.closedText)

class WpDexWidget(QtWidgets.QFrame, Adaptor):
	"""ui view for a WpDex object
	many widgets for one WpDex
	registered against WpDex type
	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()

	# UNSURE if we should make this explicitly match wpdex types,
	# rather than value types - we guarantee more this way
	forTypes = (WpDex, )
	dispatchInit = False # no automatic dispatching

	def __init__(self, dex:WpDex, parent=None):
		super(WpDexWidget, self).__init__(parent=parent)
		self._dex = dex
		self.childWidgets : dict[str, WpDexWidget] = {}
		# unsure if it's worth enforcing this registration
		# on random child helper widgets -
		# if you need them in more than one function,
		# add them to the map
		self.extraWidgets : dict[str, QtWidgets.QWidget] = {}
		self.buildChildWidgets()
		self.buildExtraWidgets()
		self.buildLayout()
		# set border
		#self.setStyleSheet("border: 1px solid black;")

		self.setAppearance()



	# def postInit(self):
	# 	"""test as a manually-called post-init method to build widgets -
	# 	this lets us override init, but that itself is kind of pointless
	# 	until child widgets get built"""
	# 	self.buildChildWidgets()
	# 	self.buildExtraWidgets()

	def dex(self)->WpDex:
		return self._dex

	def getOverride(self, match:dict,
	                baseValue:T.Any)->T.Any:
		"""PLACEHOLDER
		for now just return base value -
		connect to common override logic once ready
		"""
		return baseValue

	def childWidgetType(self, child:WpDex)->T.Type[WpDexWidget]:
		"""return widget type for given child -
		check if any overrides are specified for this path, object type
		etc"""
		baseWidgetType = self.adaptorForObject(child)
		# check for overrides
		finalWidgetType = self.getOverride(
			match={"purpose" : "widgetType",
			       "path" : child.path,
			       "obj": child},
			baseValue=baseWidgetType
		)
		return finalWidgetType

	# init and build methods
	def buildChildWidgets(self):
		"""populate childWidgets map with widgets
		for all dex children"""
		dex = self.dex()
		log("buildChildWidgets")
		for key, child in self.dex().children.items():
			childWidgetType = self.childWidgetType(child)
			childWidget = childWidgetType(child, parent=self)
			self.childWidgets[key] = childWidget

	def buildExtraWidgets(self):
		#self.extraWidgets["typeLabel"] = WpTypeLabel(parent=self)
		self.typeLabel = WpTypeLabel(parent=self)
		pass
	def buildLayout(self):
		raise NotImplementedError
		# if self.childWidgets:
		# 	layout = QtWidgets.QVBoxLayout(self)
		# 	for key, widget in self.childWidgets.items():
		# 		layout.addWidget(widget)
		# 	self.setLayout(layout)
		# else:
		# 	self.typeLabel.hide()

	def setAppearance(self):
		"""runs after all setup steps"""

	# others
	def uiPath(self)->list[str]:
		"""return path to this widget in the ui
		lol wouldn't it be mental if this could diverge from the
		wpdex path
		but like actually though"""
		return self.dex().path



# main widget
class WpDexWindow(QtWidgets.QWidget, Adaptor):
	"""top-level widget to display a superItem view -
	single embed in a host ui"""

	def __init__(self, parent=None, rootObj:WpDex=None):
		super(WpDexWindow, self).__init__(parent=parent)
		self._dex = None
		self.itemWidget : WpDexWidget = None

		self.emptyLabel = QtWidgets.QLabel("NO WPDEX PROVIDED, WINDOW EMPTY",
		                                   parent=self)
		layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.emptyLabel)
		self.setLayout(layout)
		#self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
		#self.layout : QtWidgets.QLayout = None
		if rootObj is not None:
			self.setRootObj(rootObj)


	def dex(self)->WpDex:
		return self._dex

	def setRootObj(self, obj:WpDex):
		"""set top level, sync everything below"""
		if not isinstance(obj, WpDex):
			obj = WpDex(obj)
		self._dex = obj
		self.sync()

	def sync(self):
		"""rebuild all contained widgets"""
		self.emptyLabel.hide()
		if self.itemWidget:
			self.itemWidget.hide()
			self.itemWidget.deleteLater()
			self.itemWidget = None
		if not self.dex():
			self.emptyLabel.show()
			return

		self.itemWidget = WpDexWidget.adaptorForObject(self.dex())(self.dex(), parent=self)
		self.layout().addWidget(self.itemWidget)



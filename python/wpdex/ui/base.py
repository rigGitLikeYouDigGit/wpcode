


from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from dataclasses import dataclass

#import pretty_errors

from wplib import log
from wplib.object import Adaptor, VisitAdaptor
from wplib.serial import SerialAdaptor, Serialisable, serialise, deserialise

from wpdex import WpDex, WpDexProxy, getWpDex, rx, WX
from wpdex.ui.atomic import AtomicWidget

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

	def __init__(self, parent:WpDexView=None,
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
		def parent(self)->WpDexView:
			"""return parent widget"""
			return self.parent()

	def setOpenState(self, state:bool):
		"""set open state of label"""
		if state:
			self.setText(self.openText)
		else:
			self.setText(self.closedText)

class WpDexView(AtomicWidget):
	"""ui view for a WpDex object
	contains child widgets in a grid-like spreadsheet tree view
	each view contains only single layer of widgets
	many widgets for one WpDex
	registered against WpDex type
	"""
	# adaptorTypeMap = Adaptor.makeNewTypeMap()
	#
	# # UNSURE if we should make this explicitly match wpdex types,
	# # rather than value types - we guarantee more this way
	# forTypes = (WpDex, )
	# dispatchInit = False # no automatic dispatching

	# def __init__(self, value, parent=None):
	# 	QtWidgets.QAbstractItemView.__init__(self, parent)
	# 	AtomicWidget.__init__(self, value)
	#
	# 	self.childWidgets : dict[WpDex.pathT, WpDexView] = {}
	# 	# unsure if it's worth enforcing this registration
	# 	# on random child helper widgets -
	# 	# if you need them in more than one function,
	# 	# add them to the map
	# 	self.extraWidgets : dict[str, QtWidgets.QWidget] = {}
	# 	self.buildChildWidgets()
	# 	self.buildExtraWidgets()
	# 	self.buildLayout()
	# 	# set border
	# 	#self.setStyleSheet("border: 1px solid black;")
	#
	# 	self.setAppearance()
	#
	#
	#
	#
	# # def getOverride(self, match:dict,
	# #                 baseValue:T.Any)->T.Any:
	# # 	"""PLACEHOLDER
	# # 	for now just return base value -
	# # 	connect to common override logic once ready
	# # 	"""
	# # 	return baseValue
	# #
	# # def childWidgetType(self, child:WpDex)->T.Type[WpDexView]:
	# # 	"""return widget type for given child -
	# # 	check if any overrides are specified for this path, object type
	# # 	etc"""
	# # 	baseWidgetType = self.adaptorForObject(child)
	# # 	# check for overrides
	# # 	finalWidgetType = self.getOverride(
	# # 		match={"purpose" : "widgetType",
	# # 		       "path" : child.path,
	# # 		       "obj": child},
	# # 		baseValue=baseWidgetType
	# # 	)
	# # 	return finalWidgetType
	#
	# # init and build methods
	# def buildChildWidgets(self):
	# 	"""populate childWidgets map with widgets
	# 	for all dex children"""
	# 	dex = self.dex()
	# 	log("buildChildWidgets")
	# 	for key, child in self.dex().children.items():
	# 		childWidgetType = self.childWidgetType(child)
	# 		childWidget = childWidgetType(child, parent=self)
	# 		self.childWidgets[key] = childWidget
	#
	# def buildExtraWidgets(self):
	# 	#self.extraWidgets["typeLabel"] = WpTypeLabel(parent=self)
	# 	self.typeLabel = WpTypeLabel(parent=self)
	# 	pass
	# def buildLayout(self):
	# 	raise NotImplementedError
	# 	# if self.childWidgets:
	# 	# 	layout = QtWidgets.QVBoxLayout(self)
	# 	# 	for key, widget in self.childWidgets.items():
	# 	# 		layout.addWidget(widget)
	# 	# 	self.setLayout(layout)
	# 	# else:
	# 	# 	self.typeLabel.hide()
	#
	# def setAppearance(self):
	# 	"""runs after all setup steps"""
	#
	# # others
	# def uiPath(self)->list[str]:
	# 	"""return path to this widget in the ui
	# 	lol wouldn't it be mental if this could diverge from the
	# 	wpdex path
	# 	but like actually though"""
	# 	return self.dex().path



# main widget
class WpDexWindow(
	#AtomicWidget,
	QtWidgets.QWidget,
	#Adaptor
                  ):
	"""top-level widget to display a superItem view -
	single embed in a host ui

	NECESSARY in case the type of the root data changes -
	need to get a new type of WpDexView widget while maintaining
	any external references to this one
	doesn't hold any value itself

	TODO: we still don't have a uniform way of getting a widget for
		any possible value - this WpDexWindow assumes that we won't try and set
		to a primitive value, always expects child dex items
		for now live with it
	"""

	def __init__(self, parent=None, value:(WpDexProxy, WX)=None):
		QtWidgets.QWidget.__init__(self, parent=parent)

		#self._dex : WpDex = None
		self.itemWidget : AtomicWidget = None

		# self.emptyLabel = QtWidgets.QLabel("NO WPDEX PROVIDED, WINDOW EMPTY",
		#                                    parent=self)
		layout = QtWidgets.QVBoxLayout(self)
		#layout.addWidget(self.emptyLabel)
		self.setLayout(layout)

		self.setContentsMargins(0, 0, 0, 0)
		self.layout().setContentsMargins(0, 0, 0, 0 )
		#AtomicWidget.__init__(self, value=value)
		self.setValue(value)

	def view(self)->WpDexView:
		return self.itemWidget

	# def _rawUiValue(self):
	# 	"""return raw result from ui, without any processing
	# 	so for a lineEdit, just text()"""
	# 	raise NotImplementedError(self)
	#
	# def _setRawUiValue(self, value):
	# 	"""set the final value of the ui to the exact value passed in,
	# 	this runs after any pretty formatting"""
	# 	raise NotImplementedError(self)

	# def dex(self)->WpDex:
	# 	return self.view().dex()

	def setValue(self, value):
		""""""
		#super()._commitValue(value)
		dex = getWpDex(value) or WpDex(value)
		#self._dex = dex
		if self.itemWidget:
			self.itemWidget.hide()
			self.layout().removeWidget(self.itemWidget)
			self.itemWidget.setParent(None)
			self.itemWidget.deleteLater()

		itemWidgetCls : type[AtomicWidget] = WpDexView.adaptorForObject(dex)
		log("item cls", itemWidgetCls)
		self.itemWidget = itemWidgetCls(
			#parent=self,
			parent=None,
             value=value )
		log("built item widget", self.itemWidget)
		self.layout().addWidget(self.itemWidget)






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

	def __init__(self, value,):
		AtomicWidget.__init__(self, value=value,
		                      )

		# set up header and general size behaviour for views -
		# override this to fine-tune it

		self.header().setDefaultSectionSize(2)
		#self.header().setMinimumSectionSize(-1) # sets to font metrics, still buffer around it
		self.header().setMinimumSectionSize(15)
		self.header().setSectionResizeMode(
			self.header().ResizeToContents
		)
		self.setColumnWidth(0, 2)
		self.setIndentation(0)

		self.setSizeAdjustPolicy(
			QtWidgets.QAbstractItemView.AdjustToContents)
		self.setHeaderHidden(True)

		self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
		self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)
		self.setContentsMargins(0, 0, 0, 0)
		self.setViewportMargins(0, 0, 0, 0)

		self.setUniformRowHeights(False)


	def modelCls(self):
		raise NotImplementedError(self)

	def _modelIndexForKey(self, key:WpDex.keyT)->QtCore.QModelIndex:
		return self.model().index(int(key), 1)

	def _clearChildWidgets(self):
		"""TODO:
		clear up where this sits between base Atomic and the View subclass
		"""
		self.setModel(self.modelCls()(parent=self))

		# clear any child widgets
		ties = tuple(self._childAtomics.items())
		self._childAtomics.clear()
		for k, w in ties:
			w.setParent(None)
			w.deleteLater()


	def _setRawUiValue(self, value):
		#raise
		#log("_set raw ui")
		self.buildChildWidgets()

	def _rawUiValue(self):
		return None



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


class DexViewExpandButton(QtWidgets.QPushButton):
	"""button to show type of container when open,
	and overview of contained types when closed"""
	expanded = QtCore.Signal(bool)
	def __init__(self, openText="[", dex:WpDex=None, parent=None):
		self._isOpen = True
		self._openText = openText
		self._dex = dex
		super().__init__(openText, parent=parent)

		m = 0
		self.setContentsMargins(m, m, m, m)
		#self.setFixedSize(20, 20)
		self.setStyleSheet("padding: 1px 1px 2px 2px; text-align: left")

		self.clicked.connect(lambda : self.setExpanded(
			state=(not self.isExpanded()), emit=True))

	def getClosedText(self):
		return self._dex.getTypeSummary()

	def setExpanded(self, state=True, emit=False):
		log("setExpanded", state, emit)
		if state:
			self.setText(self._openText)
		else:
			self.setText(self.getClosedText())
		self._isOpen = state
		if emit:
			self.expanded.emit(state)
	def isExpanded(self):
		return self._isOpen

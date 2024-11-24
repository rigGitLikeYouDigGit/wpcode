


from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from dataclasses import dataclass

#import pretty_errors

from wplib import log
from wplib.object import Adaptor, VisitAdaptor
from wplib.serial import SerialAdaptor, Serialisable, serialise, deserialise

from wpdex import WpDex, WpDexProxy, getWpDex, rx, WX
from wpdex.ui.atomic import AtomicWidgetOld

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


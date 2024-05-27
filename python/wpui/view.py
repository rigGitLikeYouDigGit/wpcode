
from __future__ import annotations
import typing as T


from PySide2 import QtCore, QtWidgets, QtGui

"""this module represents weeks of sporadic misery"""

def syncViewLayout(view:QtWidgets.QAbstractItemView):
	view.scheduleDelayedItemsLayout()
	view.executeDelayedItemsLayout()

class IndexWidgetHolder(QtWidgets.QWidget):
	"""
	CALLS setParent() on inner widget, parents it to this one

	widgets set directly as index widgets cannot use
	layouts, or spacing, so we need a tiny simple holder widget
	to contain them


	you have NO IDEA how long it took me to figure that out


	"""

	def __init__(self, parent=None, innerWidget:QtWidgets.QWidget=None,
	             fillBackground=True):
		super().__init__(parent)
		self.innerWidget = innerWidget
		self.innerWidget.setParent(self)
		if fillBackground:
			self.setAutoFillBackground(True)

	def sizeHint(self):
		"""this should get read properly by item view,
		when used with the function above"""
		return self.innerWidget.sizeHint()


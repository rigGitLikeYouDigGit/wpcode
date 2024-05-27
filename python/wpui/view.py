
from __future__ import annotations
import typing as T


from PySide2 import QtCore, QtWidgets, QtGui


def syncViewLayout(view:QtWidgets.QAbstractItemView):
	view.scheduleDelayedItemsLayout()
	view.executeDelayedItemsLayout()



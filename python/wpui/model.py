
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets


def iterAllItems(item:QtGui.QStandardItem=None,
                   model:QtGui.QStandardItemModel=None )->T.Iterator[QtGui.QStandardItem]:
	"""iterate all indices in a model"""
	item = item or model.invisibleRootItem()
	for row in range(item.rowCount()):
		for col in range(item.columnCount()):
			yield from iterAllItems(item=item.child(row, col))
	yield item




from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel

if T.TYPE_CHECKING:
	from wpui.superitem.combined import SuperItemBase


class SuperModel(QtGui.QStandardItemModel):
	"""not the photographic kind, this is way better"""

	def __init__(self):
		super(SuperModel, self).__init__()
		self.parentSuperItem : SuperItemBase = None

	def data(self, index:QtCore.QModelIndex, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		return super(SuperModel, self).data(index, role)

	def headerData(self, section:int, orientation:QtCore.Qt.Orientation, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		if role == QtCore.Qt.FontRole:
			f = QtGui.QFont()
			f.setPointSize(6)
			return f
		return super(SuperModel, self).headerData(section, orientation, role)
	#



from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginBase

from wpui.superitem.model import SuperModel
from wpui.superitem.delegate import SuperDelegate
from wpui.superitem.plugin import SuperItemPlugin
from wpui.superitem.item import SuperItem
from wpui.superitem.view import SuperViewBase

"""provide plugin adaptors for stl containers - 
dictionaries and lists
"""

class SuperListView(SuperViewBase, QtWidgets.QListView):
	def __init__(self, *args, **kwargs):
		QtWidgets.QListView.__init__(self, *args, **kwargs)
		SuperViewBase.__init__(self, *args, **kwargs)


class ListSuperItemPlugin(SuperItemPlugin):
	viewCls = SuperListView



class SuperTableView(SuperViewBase, QtWidgets.QTableView):
	def __init__(self, *args, **kwargs):
		QtWidgets.QTableView.__init__(self, *args, **kwargs)
		SuperViewBase.__init__(self, *args, **kwargs)

		# header = self.horizontalHeader()
		#self.horizontalHeader().setFirstSectionMovable(False)
		self.horizontalHeader().setCascadingSectionResizes(True)
		self.horizontalHeader().setStretchLastSection(True)
		# self.setHorizontalHeader(header)

		# self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
		# self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


	def createOwnChildItems(self, value)->list[SuperItem]:
		rows = []
		if isinstance(value, MAP_TYPES):
			for i in value.items():
				rows.append(
					[self.createChildItem(s) for s in i]
					#[self.createChildItem(i[0]), self.createChildItem(i[1])]
				)
			return rows
		if isinstance(value, SEQ_TYPES):
			#rows.append(self.createChildItem(value))
			for i in value:
				rows.append(self.createChildItem(i))
			return rows
		raise TypeError(f"value {value} must be a sequence or mapping")



class DictSuperItemPlugin(SuperItemPlugin):
	viewCls = SuperTableView
from __future__ import annotations

import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wplib import log

from wplib.constant import LITERAL_TYPES
from wpui.typelabel import TypeLabel
from wpui.superitem import SuperItem, SuperItemWidget, SuperItemView, SuperModel, SuperModelParams

class LiteralSuperItem(SuperItem):
	"""model for a literal"""
	forTypes = LITERAL_TYPES

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		return self.wpPyObj


class LiteralSuperItemWidget(SuperItemWidget):
	"""widget for a literal -
	for now just a text box

	augment with a type label to keep track of a consistent type,
	or to change type if needed
	"""
	forTypes = LITERAL_TYPES
	def __init__(self, superItem:LiteralSuperItem, parent=None,
	             expanded=True):
		super().__init__( superItem, parent=parent)
		#log(f"LiteralSuperItemWidget.__init__({superItem}, {parent})")
		#self.setAutoFillBackground(True)
		#return

		#self.typeLabel = QtWidgets.QLabel(str(type(superItem.wpPyObj)), parent=self)
		# self.typeLabel = TypeLabel.forObj(superItem.wpPyObj, parent=self)

		self.text = QtWidgets.QLineEdit(str(superItem.wpPyObj), parent=self)
		self.text.textEdited.connect(self._onTextEdited)

		self.makeLayout()


	def makeLayout(self):
		layout = QtWidgets.QHBoxLayout()
		#layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.typeLabel)
		layout.addWidget(self.text)
		self.setLayout(layout)
		#self.layout().setContentsMargins(3, 3, 3, 3)


	def _onTextEdited(self, *args, **kwargs):
		"""update the pyObj on text edit"""
		log(f"LiteralSuperItemWidget._onTextEdited({args}, {kwargs})")
		self.superItem.wpPyObj = self.text.text()

	def sizeHint(self):
		"""return the size hint"""
		#print("w size hint")
		#return QtCore.QSize(100, 100)
		return self.contentsRect().size()



# class StringSuperItem(SuperItem):
# 	"""not sure how the split between string and literal might work
# 	"""
# 	forTypes = [str]
#
# 	def wpResultObj(self) ->T.Any:
# 		"""return the result object"""
# 		return self.wpPyObj

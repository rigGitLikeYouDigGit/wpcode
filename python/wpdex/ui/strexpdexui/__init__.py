
"""string and expression widget resources -
split into files later"""


from __future__ import annotations

import typing as T

from PySide2 import QtWidgets

from wplib import log

from wplib.constant import LITERAL_TYPES
from wpdex.ui.base import WpDexItem, WpDexWindow


class StrDexItem(WpDexItem):
	"""model for string - nothing needed by default,
	delegate to fancier widgets if needed"""
	forTypes = (str, )

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		return self.wpPyObj


class StrDexWidget(WpDexWindow):
	"""widget for string
	"""
	forTypes = LITERAL_TYPES
	def __init__(self, superItem:StrDexItem, parent=None,
	             expanded=True):
		super().__init__( superItem, parent=parent)
		#log(f"PrimDexWidget.__init__({superItem}, {parent})")
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
		log(f"PrimDexWidget._onTextEdited({args}, {kwargs})")
		self.superItem.wpPyObj = self.text.text()

	def sizeHint(self):
		"""return the size hint"""
		#print("w size hint")
		#return QtCore.QSize(100, 100)
		return self.contentsRect().size()



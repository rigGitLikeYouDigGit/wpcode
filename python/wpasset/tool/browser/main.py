from __future__ import annotations


import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wp.treefield import TreeField, TreeFieldParams
from wp import option, constant

"""show what assets are available in the project"""

class AssetBrowserWidget(QtWidgets.QWidget):
	pass
#
# 	def __init__(self, parent: T.Optional[QtWidgets.QWidget] = None):
# 		super(AssetBrowserWidget, self).__init__(parent=parent)
#
# 		self.searchWidget = AssetSearchWidget()
# 		self.assetListWidget = AssetListWidget()
#
# 		self.mainLayout = QtWidgets.QVBoxLayout()
# 		self.mainLayout.addWidget(self.searchWidget)
# 		self.mainLayout.addWidget(self.assetListWidget)
#
# 		self.setLayout(self.mainLayout)


if __name__ == '__main__':
	import sys
	from PySide2.QtWidgets import QApplication

	app = QApplication(sys.argv)
	widget = AssetBrowserWidget()
	widget.show()
	sys.exit(app.exec_())


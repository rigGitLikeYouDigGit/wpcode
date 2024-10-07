from __future__ import annotations

from PySide2 import QtWidgets

from wpui import layout
from wp.ui.widget import ChildListWidget





class AssetBrowserWidget(QtWidgets.QWidget):
	pass



if __name__ == '__main__':
	import sys
	from PySide2.QtWidgets import QApplication

	app = QApplication(sys.argv)
	widget = AssetBrowserWidget()
	widget.show()
	sys.exit(app.exec_())


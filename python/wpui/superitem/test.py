



import sys

from PySide2 import QtCore, QtGui, QtWidgets


from wpui.layout import genAutoLayout
from wpui.superitem import SuperItem


if __name__ == '__main__':
	import sys
	import qt_material
	app = QtWidgets.QApplication(sys.argv)
	qt_material.apply_stylesheet(app, theme='dark_blue.xml')

	#model = SuperModel([1, 2, 3])
	data = [1, 2, 3]
	item = SuperItem.forData(data)

	sys.exit(app.exec_())

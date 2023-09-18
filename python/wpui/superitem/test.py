



import sys

from PySide2 import QtCore, QtGui, QtWidgets

from wplib.object import visit

from wpui.layout import genAutoLayout
from wpui.superitem import SuperItem


if __name__ == '__main__':
	import sys

	structure = {
		"root": {
			"a": 1,

			"b": (2, 3, {"key" : "asdaj"}, (333, 444,)),
			# "branch": {
			# 	(2, 4) : [1, {"key" : "val"}, "chips", 3],
			# }
		},
		"root2": {
			"a": 1,
		}
	}

	app = QtWidgets.QApplication([])

	item = SuperItem.forValue(structure)
	w = item.getNewView()


	w.show()
	sys.exit(app.exec_())

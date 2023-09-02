



import sys

from PySide2 import QtCore, QtGui, QtWidgets

from wplib.object import visit

from wpui.layout import genAutoLayout



if __name__ == '__main__':

	structure = {
		"root": {
			"a": 1,
			"b": (2, 3),
			"branch": {
				(2, 4) : [1, {"key" : "val"}, "chips", 3],
			}
		}
	}

	app = QtWidgets.QApplication([])

	w = QtWidgets.QWidget()
	widget = TreeWidget(tree=tree, parent=w)
	btn = QtWidgets.QPushButton("sync tree", parent=w)
	btn.clicked.connect(lambda *args, **kwargs: widget.setTree(tree))

	w.setLayout(genAutoLayout(w, recurse=False))



	w.show()
	sys.exit(app.exec_())





from PySide2 import QtWidgets, QtCore, QtGui

from tree.dev.log import logfn, logwidget, testmodule

from tree.dev.log.testmodule import mainTestFn, loopTestFn

if __name__ == '__main__':


	# loopTestFn()


	import sys

	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()


	widg = logwidget.TreeLogWidget(
		parent=None,
		tree=logfn.logTree)


	win.setCentralWidget(widg)
	win.show()
	loopTestFn()

	# mainTestFn()
	#
	# #app.exec_()
	#
	#
	sys.exit(app.exec_())


	pass

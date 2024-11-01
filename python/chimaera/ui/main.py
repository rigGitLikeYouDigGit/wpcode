

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wptree import Tree
from wpdex import WpDexProxy
from wpdex.ui.atomic import AtomicWidget
from chimaera.node import ChimaeraNode
from chimaera.ui.scene import ChimaeraScene
from chimaera.ui.view import ChimaeraView
from chimaera.ui.node import NodeDelegate


class ChimaeraWidget(QtWidgets.QWidget, AtomicWidget
                     ):

	def __init__(self, value:ChimaeraNode=None, parent=None):
		QtWidgets.QWidget.__init__(self, parent=parent)
		AtomicWidget.__init__(self, value=value)
		self.scene = ChimaeraScene(
			graph=self.rxValue(),
			parent=self,
		                           )
		self.view = ChimaeraView(scene=self.scene, parent=self)
		self.scene.setGraph(self.scene.graph()) #TODO: cringe forcing graph rxs to fire
		layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.view)
		self.setFocusPolicy(QtCore.Qt.NoFocus)

	# def focusNextPrevChild(self, next):
	# 	return False
	#
	# def nextInFocusChain(self):
	# 	return self
	#
	# def focusOutEvent(self, event:QtGui.QFocusEvent):
	# 	return


def printFn(*args, **kwargs):
	log("WATCHED", args, kwargs)

def showW():

	w = ChimaeraView(scene=ChimaeraScene())
	w.show()
	node = ChimaeraNode.create(name="testNode")
	delegate = NodeDelegate(node)
	#log("del", delegate)
	#log(delegate.nameLine.text())
	w.scene().addItem(delegate)
	return w

if __name__ == '__main__':
	app = QtWidgets.QApplication()
	w = showW()
	app.exec_()

	# from param.reactive import rx
	# from PySide6 import QtWidgets # or PySide6, I get the same result
	#
	# rxtext = rx("")
	#
	# # create the Qt application, required to start building widgets
	# # event loop doesn't run yet
	# app = QtWidgets.QApplication()
	# line = QtWidgets.QLineEdit() # create a QLineEdit widget instance
	#
	# rxtext.rx.watch(line.setText)
	# rxtext.rx.value="some-value"
	#
	#
	# # display the widget
	# line.show()
	# # run the Qt application
	# app.exec()

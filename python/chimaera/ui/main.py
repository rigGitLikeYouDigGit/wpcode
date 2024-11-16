

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log, inheritance
from wptree import Tree
from wpdex import WpDexProxy
from wpdex.ui.atomic import AtomicWidgetOld, AtomicUiInterface
from chimaera.node import ChimaeraNode
from chimaera.ui.scene import ChimaeraScene
from chimaera.ui.view import ChimaeraView
from chimaera.ui.node import NodeDelegate


class ChimaeraWidget(
	QtWidgets.QWidget,# AtomicWidgetOld,
	metaclass=inheritance.resolveInheritedMetaClass(QtWidgets.QWidget,# AtomicWidgetOld
	                                                )
                     ):
	"""for all signals and events defer to the ChimaeraScene
	as the main manager for all changes?
	"""

	def __init__(self, value:ChimaeraNode=None, parent=None):
		QtWidgets.QWidget.__init__(self, parent=parent)
		assert value is not None, "Must pass top graph to ChimaeraWidget"
		self.scene = ChimaeraScene(
			graph=value,
			parent=self,
		                           )
		#AtomicWidgetOld.__init__(self, value=None)
		self.view = ChimaeraView(scene=self.scene, parent=self)
		#self.scene.setGraph(self.scene.graph()) #TODO: cringe forcing graph rxs to fire
		layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.view)
		self.setFocusPolicy(QtCore.Qt.NoFocus)

	def value(self):
		return self.scene.graph()
	def rxValue(self):
		return self.scene.rxGraph()

	def _rawUiValue(self):
		"""all edits to chimaera scene are applied immediately -
		we lose the ability to drag nodes live, but for now that is ok
		"""
		return self.scene.graph()
	def _setRawUiValue(self, value):
		pass

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

	graph = ChimaeraNode.create("graphNode")
	p = WpDexProxy(graph)
	s = ChimaeraScene(p.ref())
	node = p.createNode(ChimaeraNode, name="childNode")

	w = ChimaeraView(scene=s)
	w.show()
	#node = ChimaeraNode.create(name="testNode")
	# delegate = NodeDelegate(node)
	# log("del", delegate)
	# log(delegate.nameLine.text())
	# w.scene().addItem(delegate)
	# print("")
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

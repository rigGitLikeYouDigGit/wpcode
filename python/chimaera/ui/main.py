

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wptree import Tree
from wpdex import WpDexProxy
from chimaera.node import ChimaeraNode
from chimaera.ui.scene import ChimaeraScene
from chimaera.ui.view import ChimaeraView
from chimaera.ui.node import NodeDelegate, ReactLineEdit


class ChimaeraWidget(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent=parent)
		self.scene = ChimaeraScene(parent=self)
		self.view = ChimaeraView(scene=self.scene, parent=self)
		layout = QtWidgets.QVBoxLayout

def printFn(*args, **kwargs):
	log("WATCHED", args, kwargs)

def showW():

	# w = ChimaeraView(scene=ChimaeraScene())
	# w.show()
	# node = ChimaeraNode.create(name="testNode")
	# delegate = NodeDelegate(node)
	# log("del", delegate)
	# #log(delegate.nameLine.text())
	# w.scene().addItem(NodeDelegate(node))
	# return w
	from param import rx, bind

	t = Tree("testTree")
	p = WpDexProxy(t)
	ref = p.ref("@N")
	# log("ref", ref, ref.rx.value)
	# p.name = "2ND NAME"
	# log("after rename", ref, ref.rx.value)
	w = ReactLineEdit(text=ref)

	#printFn = lambda *a, **kwargs: log("WATCHED", a, kwargs)
	#ref.rx.watch()
	ref.rx.watch(fn=(printFn),
	             onlychanged=True,
	             #queued=True,
	             #precedence=1
	             )

	"""onlychanged just isn't passed through to the watcher machinery in rx,
	and there's some weird treatment of kwargs that stops me from adding it easily.
	going with a hacky dirty argument in the ref for now
	"""

	# bind(printFn,
	#      ref.rx._reactive,
	#      #ref.rx,
	#      watch=True)

	#p.name = "3RD NAME"
	# from param.display import _display_accessors, _reactive_display_objs
	# print("final display accessors", _display_accessors)
	# print("final display objs", _reactive_display_objs)

	#ref.WRITE("eyyyyy")

	#raise
	w.show()
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

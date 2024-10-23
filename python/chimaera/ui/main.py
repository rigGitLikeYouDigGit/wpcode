

from PySide2 import QtCore, QtWidgets, QtGui


from chimaera.node import ChimaeraNode
from chimaera.ui.scene import ChimaeraScene
from chimaera.ui.view import ChimaeraView
from chimaera.ui.node import NodeDelegate


class ChimaeraWidget(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent=parent)
		self.scene = ChimaeraScene(parent=self)
		self.view = ChimaeraView(scene=self.scene, parent=self)
		layout = QtWidgets.QVBoxLayout


def showW():

	w = ChimaeraView(scene=ChimaeraScene())
	w.show()
	node = ChimaeraNode.create(name="testNode")
	w.scene().addItem(NodeDelegate(node))
	return w


if __name__ == '__main__':
	app = QtWidgets.QApplication()
	w = showW()
	app.exec_()



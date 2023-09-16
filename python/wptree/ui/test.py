
import sys

from PySide2 import QtCore, QtGui, QtWidgets

from wpui.layout import genAutoLayout

from wptree import Tree
from wptree.ui.view import TreeView
from wptree.ui.model import TreeModel
from wptree.ui.widget import TreeWidget



if __name__ == '__main__':

	tree = Tree("root")
	tree.lookupCreate = True
	tree["a"] = 1
	tree["b"] = (2, 3)
	tree["branch", "leaf"] = "leaf value"
	tree["branch", "branch2", "leaf2"] = {(1, 2): "leaf2 value",
	                                      "tree": Tree("leaf2 tree value")
	                                      }
	tree["branch", "branch2", "leaf3"] = "leaf3 value"
	tree["branch", "leaf4"] = "leaf4 value"


	app = QtWidgets.QApplication([])

	w = QtWidgets.QWidget()
	widget = TreeWidget(tree=tree, parent=w)
	btn = QtWidgets.QPushButton("sync tree", parent=w)
	btn.clicked.connect(lambda *args, **kwargs: widget.setTree(tree))

	w.setLayout(genAutoLayout(w, recurse=False))



	w.show()
	sys.exit(app.exec_())


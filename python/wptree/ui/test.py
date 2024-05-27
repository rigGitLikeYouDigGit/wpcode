
import sys

from PySide2 import QtWidgets

from wptree import Tree

from wpdex.ui.superitem import SuperItem

if __name__ == '__main__':

	valTree = Tree("valRoot")
	valTree.lookupCreate = True
	valTree["a", "b"] = {"a": 1}

	tree = Tree("root")
	tree.lookupCreate = True
	#tree.value = {"a": 1}

	tree["b"] = {"a": 1,
	             "b": valTree,
	             "c": 3,}
	tree["a"] = 1

	tree["branch", "leaf"] = "leaf value"
	tree["branch", "branch2", "leaf2"] = { "a": 1,
	                                      "b": 2,
		(1, 2): "leaf2 value",
	                                      "tree": Tree("leaf2 tree value")
	                                      }
	tree["branch", "branch2", "leaf3"] = "leaf3 value"
	tree["branch", "leaf4"] = "leaf4 value"
	tree["b"] = (2, 3)

	tree["c"] = [1, [1, 2], 2]

	app = QtWidgets.QApplication([])

	item = SuperItem.forValue(tree)
	w = item.getNewView()
	w.show()
	sys.exit(app.exec_())

	#w = QtWidgets.QWidget()
	#widget = TreeWidget(tree=tree, parent=w)
	#btn = QtWidgets.QPushButton("sync tree", parent=w)
	#btn.clicked.connect(lambda *args, **kwargs: widget.setTree(tree))

	#w.setLayout(genAutoLayout(w, recurse=False))



	# w.show()
	# sys.exit(app.exec_())
	#

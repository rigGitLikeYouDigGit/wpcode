
import os, sys


import unittest

from wptree import Tree

from PySide2 import QtWidgets

from wpdex.ui.superitem import SuperItem
from wpdex.ui.superitem import TreeSuperItem



jsonOutPath = os.path.sep.join(
	os.path.split(__file__ )[:-1]) + "testLog.json"

class CustomTreeType(Tree):
	pass

class TestMainTree(unittest.TestCase):
	""" test for main tree interface methods """

	def test_itemTypeRetrieval(self):
		pass

	def test_items(self):
		app = QtWidgets.QApplication(sys.argv)
		data = [1, 2, 3]
		item = SuperItem.forData(data)
		self.assertFalse(data is item.wpResultObj())
		self.assertEqual(str(data), str(item.wpResultObj()))

		data = {"a": 1, "b": [6, {"b": "d"}], "c": 3}
		item = SuperItem.forData(data)
		print("data", data)
		print("item", item)
		print("result", item.wpResultObj())
		app.quit()
		#app.exec_()

	def test_treeItem(self):
		app = QtWidgets.QApplication(sys.argv)
		t = Tree("root", 55)
		t("branchA", create=True).value = "first branch"
		t("branchA", create=True)("leafA", create=True).value = "first leaf"
		t("branchB", create=True).value = 2
		item = SuperItem.forData(t)

		self.assertIsInstance(item, SuperItem)
		self.assertIsInstance(item, TreeSuperItem)

		# recover the tree from the item
		newTree = item.wpResultObj()
		self.assertIsInstance(newTree, Tree)
		self.assertEqual(newTree, t)
		self.assertFalse(newTree is t)
		self.assertEqual(newTree("branchA").value, "first branch")
		self.assertFalse(newTree("branchA") is t("branchA"))



		app.quit()


if __name__ == '__main__':

	unittest.main()




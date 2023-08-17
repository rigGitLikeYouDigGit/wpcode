
from __future__ import print_function

from sys import version_info
import os

import unittest

from wptree.main import Tree
from tree.lib.proxy import TreeProxy
from wptree.reference import TreeReference

from tree.test.constant import midTree


class TestTreeProxy(unittest.TestCase):
	""" test for direct proxies of trees """

	def setUp(self):
		""" construct a basic test tree """

		self.tree = midTree.copy()

	# def test_weakSet(self):
	# 	weak = WeakValueDictionary()
	# 	item = ["test"]

	def test_treeReference(self):
		""" test that a TreeReference object remains valid
		through deletion and recreation, by uid"""
		testBranch = self.tree("branchA")
		testUid = testBranch.uid
		ref = TreeReference(testBranch)
		self.assertIs(ref.resolve(), testBranch)

		# delete
		testBranch.remove()
		testBranch.delete()

		self.assertRaises(KeyError, ref.resolve)

		# recreate
		newBranch = self.tree("newBranch")
		newBranch.setUid(testUid)

		self.assertIs(ref.resolve(), newBranch)

	def test_treeUidSeeding(self):
		""" test that a Tree object uid can be regenerated consistently"""
		testBranch = self.tree("branchA")
		copyBranch = Tree("newBranchA")
		copyBranch.setAuxProperty("_baseUid", "eyyyyy")
		baseCopyUid = copyBranch.uid

		testBranch.seedUids(copyBranch)
		seededUid = copyBranch.uid

		self.assertNotEqual(baseCopyUid, seededUid)

		newRef = TreeReference(copyBranch)

		# regenerate a new tree
		secondCopyBranch = Tree("newBranchB")
		secondCopyBranch.setAuxProperty("_baseUid", "eyyyyy")

		testBranch.seedUids(secondCopyBranch)

		self.assertEqual(secondCopyBranch.uid, seededUid)
		self.assertIs(newRef.resolve(), secondCopyBranch)

	def test_treeProxy(self):
		testBranch = self.tree("branchA")
		proxy = TreeProxy.getProxy(testBranch)

		self.assertIsNot(testBranch, proxy)
		self.assertIsInstance(proxy, Tree)

		proxyLookup = proxy("leafA")






if __name__ == '__main__':

	unittest.main()




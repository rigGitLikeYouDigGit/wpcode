
from __future__ import print_function

import unittest

from tree import Tree
from wptree.delta import TreeNameDelta, TreeCreationDelta

from tree.test.constant import midTree

class TestTreeReference(unittest.TestCase):
	""" test for main tree interface methods """

	def setUp(self):
		""" construct a basic test tree """

		self.tree = midTree.copy()

	def test_sameTreeRefCreation(self):
		""" test that reference to branch within same tree can
		 be created"""
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		proxyBranch = refParentBranch.addReferenceBranch(refTargetBranch)
		self.assertIsNot(refTargetBranch, proxyBranch,
		                 """proxy branch is literally target branch""")

	def test_treeRefParenting(self):
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		proxyBranch = refParentBranch.addReferenceBranch(refTargetBranch)
		self.assertEqual(proxyBranch.name, refTargetBranch.name,
		                 msg="Proxy branch does not match target branch name")
		self.assertIs(proxyBranch.parent, refParentBranch)
		self.assertIsNot(refTargetBranch.parent, proxyBranch.parent,
		                 msg="Proxy parent is literally target branch parent")

	def test_treeRefAddress(self):
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		proxyBranch = refParentBranch.addReferenceBranch(refTargetBranch)
		self.assertEqual(proxyBranch.address(), ["branchB", "branchA"])
		# print(refTargetBranch.address())
		# print(proxyBranch.address())

	def test_deltas(self):
		"""test deltas without a live input"""
		refTargetBranch = self.tree("branchA")
		refTargetBranch.setTrackingDeltas(True)

		refTargetBranch.name = "newBranchName"
		self.assertEqual(refTargetBranch.name, "newBranchName")
		self.assertEqual(len(refTargetBranch.deltaTracker().deltaStack), 1)
		self.assertIsInstance(refTargetBranch.deltaTracker().deltaStack[0], TreeNameDelta)

		refTargetBranch.undo()
		self.assertEqual(refTargetBranch.name, "branchA")

		# add new branch
		deltaLeaf = refTargetBranch("deltaLeaf", create=True)
		self.assertEqual(len(refTargetBranch.deltaTracker().deltaStack), 1)
		self.assertIsInstance(refTargetBranch.deltaTracker().deltaStack[0], TreeCreationDelta)
		self.assertIs(deltaLeaf, refTargetBranch("deltaLeaf"))



	def test_persistence(self):
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		proxyBranch = refParentBranch.addReferenceBranch(refTargetBranch)

		proxyLeaf = proxyBranch("leafA")
		resultUid = proxyBranch.liveInput.resultTree()("leafA").uid
		print("")
		print("before rename")
		refTargetBranch("leafA").name = "newLeafName"
		print("after rename")
		self.assertIs(proxyBranch("newLeafName"), proxyLeaf)

		self.assertEqual(proxyBranch.liveInput.resultTree()("newLeafName").uid, resultUid)
		self.assertEqual(proxyLeaf.name, "newLeafName")


	def test_treeNameSync(self):
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		proxyBranch = refParentBranch.addReferenceBranch(refTargetBranch)
		self.assertEqual(proxyBranch.name, "branchA")
		retrieveBranch = proxyBranch("leafA", create=False)
		# print("proxy branch ", proxyBranch.uidTie(), proxyBranch.coreData())
		# print("new parent", retrieveBranch.coreData())
		self.assertIs(retrieveBranch.parent, proxyBranch)
		self.assertIs(retrieveBranch, proxyBranch("leafA"))
		#raise

		# check that a live input sets the reference branch to read only
		self.assertTrue(proxyBranch.readOnly)

		# rename source branch
		refTargetBranch.name = "newBranchName"
		self.assertEqual(proxyBranch.name, "newBranchName")

		# check retrieving branch
		retrieveBranch = proxyBranch("leafA", create=False)
		self.assertIsInstance(retrieveBranch, Tree)
		self.assertIsNot(retrieveBranch, refTargetBranch("leafA", create=False))
		self.assertIs(retrieveBranch, proxyBranch("leafA", create=False))

		# check renaming leaf
		print("before name change")
		refLeaf = refTargetBranch("leafA")
		print("")
		print("before rename")
		refLeaf.name = "leafRenamed"
		print("after rename")
		self.assertIs(proxyBranch("leafRenamed"), retrieveBranch)
		self.assertIsInstance(proxyBranch.getBranch("leafRenamed"), Tree)
		self.assertEqual("newBranchName", proxyBranch.name)

		# check that branch is now readonly
		self.assertRaises(PermissionError, proxyBranch.setName, "postLiveName")

		# test edge case of overriding readOnly - this acts as expected
		proxyBranch.readOnly = False
		proxyBranch.name = "postLiveName"
		self.assertEqual("postLiveName", proxyBranch.name)

		# test through reparenting reference branch
		self.tree("dictBranch").addBranch(refTargetBranch)

		# rename source branch after reparenting
		refTargetBranch.name = "shiftedBranchName"
		self.assertEqual(proxyBranch.name, "shiftedBranchName")

	def test_proxyReparenting(self):
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		proxyBranch = refParentBranch.addReferenceBranch(refTargetBranch)

		# add a new branch within target subtree
		refTargetAdd = refTargetBranch("refTargetAdd", create=True)
		self.assertEqual(refTargetBranch.keys(), proxyBranch.keys())


		# check that the same branch appears in reference subtree
		self.assertIsInstance(proxyBranch("refTargetAdd", create=False), Tree)

		# create a new leaf under the new branch, rename it
		newLeafName = "refTargetNewLeaf"
		newLeaf = refTargetAdd(newLeafName, create=True)
		self.assertIsInstance(proxyBranch("refTargetAdd", newLeafName,
		                                  create=False), Tree)

		proxyLeaf = proxyBranch("refTargetAdd", newLeafName)
		print(proxyLeaf, type(proxyLeaf))
		origResultUid = proxyBranch.liveInput.resultTree().uid
		origTopUid = proxyBranch.uid

		self.assertEqual(newLeaf.name, proxyLeaf.name)

		proxyParent = proxyLeaf.parent.parent

		self.assertIs(proxyParent, proxyBranch)
		self.assertNotEqual(proxyLeaf.name, "newLeafRenamed")

		print("")
		print("before rename")
		newLeaf.name = "newLeafRenamed"
		print("after rename")
		self.assertEqual(origResultUid, proxyBranch.liveInput.resultTree().uid)
		self.assertEqual(origTopUid, proxyBranch.uid)
		self.assertIs(proxyLeaf, proxyBranch("refTargetAdd", "newLeafRenamed") )
		self.assertEqual("newLeafRenamed", proxyLeaf.name)

		# check that new branch parent is top proxy branch
		self.assertIs(proxyBranch("refTargetAdd").parent,
		              proxyBranch,
		              "proxy branch parent is not correct")

		# check that live input root is the top proxy branch
		self.assertIs(proxyBranch("refTargetAdd").liveInputRoot,
		              proxyBranch)



	def test_proxyDeltas(self):
		refTargetBranch = self.tree("branchA")
		refParentBranch = self.tree("branchB")
		refBranch = refParentBranch.addReferenceBranch(refTargetBranch)


		# check name cannot be modified
		self.assertRaises(PermissionError, refBranch.setName, "testName")

		# and on branches
		print(refBranch.keys())
		self.assertRaises(PermissionError, refBranch("leafA").setName, "testName")

		# activate delta tracking
		print("")
		print("begin delta tracking")
		refBranch.setTrackingDeltas(True)
		self.assertTrue(refBranch.readOnly)
		self.assertTrue(refBranch.trackingDeltas)

		# check uv seeding in transform result tree
		baseResultUid = refBranch.liveInput.resultTree().uid

		# change name
		baseName = refBranch.name
		refBranch.name = "newRefName"

		newResultUid = refBranch.liveInput.resultTree().uid
		self.assertEqual(baseResultUid, newResultUid)
		self.assertEqual(refBranch.name, "newRefName")


		# check internal result tree is renamed
		result = refBranch.liveInput.resultTree()
		print(result.uid)
		print(refBranch.liveInput.getDeltaTransform().deltaStack)
		self.assertEqual(refBranch.liveInput.resultTree().name, "newRefName")

		# check that apparent name is changed, as delta is updated
		self.assertEqual(refBranch.name, "newRefName")
		self.assertEqual(1, len(refBranch.deltaTracker().deltaStack))

		# check undoing
		print("")
		print("before undo")
		refBranch.undo()
		print("after undo")
		self.assertEqual(refBranch.name, baseName)

		# check redoing
		refBranch.redo()
		self.assertEqual(refBranch.name, "newRefName")

		# same thing with leaf
		proxyLeaf = refBranch("leafA")
		self.assertIs(proxyLeaf.deltaTracker(), refBranch.deltaTracker())
		print(proxyLeaf.deltaTrackingRoot, refBranch.deltaTrackingRoot)
		#self.assertIs(proxyLeaf.deltaTrackingRoot, refBranch)

		self.assertEqual(proxyLeaf.name, "leafA")
		newLeafName = "newProxyLeafName"
		proxyLeaf.name = newLeafName
		self.assertEqual(proxyLeaf.name, newLeafName)
		self.assertEqual(2, len(proxyLeaf.deltaTracker().deltaStack))

		proxyLeaf.undo()
		self.assertEqual(proxyLeaf.name, "leafA")

		# test delta discarding
		proxyLeaf.name = "secondNewName"
		self.assertEqual(proxyLeaf.name, "secondNewName")
		self.assertEqual(2, len(proxyLeaf.deltaTracker().deltaStack))



		# test delta branch creation
		self.assertRaises(BaseException, proxyLeaf, "UBERLEAF", create=False)
		createdLeaf = proxyLeaf("UBERLEAF", create=True)
		createdProxyUid = createdLeaf.uid
		realLeaf = refBranch("secondNewName")
		self.assertEqual(realLeaf("UBERLEAF").uid, createdProxyUid)
		self.assertIsInstance(realLeaf("UBERLEAF"), Tree)

		self.assertIs(realLeaf("UBERLEAF"), realLeaf("UBERLEAF"))

		self.assertIs(createdLeaf, proxyLeaf("UBERLEAF", create=False))

		self.assertEqual(createdLeaf, proxyLeaf("UBERLEAF", create=False))
		self.assertEqual(createdLeaf.uid, proxyLeaf("UBERLEAF", create=False).uid)




if __name__ == '__main__':

	unittest.main()




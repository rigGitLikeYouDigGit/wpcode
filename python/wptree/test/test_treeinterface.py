import unittest

from wptree import TreeInterface, Tree

class TestMainTree(unittest.TestCase):
	""" test for main tree interface methods -
	inherit from this and override setUp to test different tree backends
	against consistent syntax"""

	treeCls = TreeInterface
	treeCls = Tree

	def setUp(self):
		""" construct a basic test tree """
		self.tree = self.treeCls(name="testRoot", value="tree root")
		# self.tree.debugOn = True
		self.tree("branchA", create=True).value = "first branch"
		self.tree("branchA", create=True)("leafA", create=True).value = "first leaf"
		self.tree("branchB", create=True).value = 2


	def test_treeInternals(self):

		baseTree = self.treeCls("basicRoot")
		self.assertEqual(baseTree.branches, [])

		print("")

		baseBranch = baseTree("basicBranch", create=True)

		print("branchmap", baseBranch.branchMap)
		self.assertTrue("basicBranch" in baseTree.branchMap)

		self.assertEqual({"basicBranch" : baseBranch}, baseTree.branchMap)
		self.assertEqual([baseBranch], baseTree.branches)



	def test_treeRoot(self):
		""" test that tree objects find their root properly """
		self.assertIs( self.tree.root, self.tree,
		                  msg="tree root is not itself")
		self.assertIs(self.tree("branchA").root, self.tree,
		                  msg="tree branch finds incorrect root of "
		                      "{}".format(self.tree))
		self.assertIs(self.tree("branchA")("leafA").root, self.tree,
		                  msg="tree leaf finds incorrect root of "
		                      "{}".format(self.tree))

		self.assertIs(self.tree("branchA", "leafA").parent, self.tree("branchA"))


	def test_treeRetrieval(self):
		""" test retrieving values and branches
		 using different methods """
		# token retrieval

		self.assertIs(self.tree("branchA", "leafA"),
		              self.tree("branchA")("leafA"),
		              msg="error in token retrieval")
		# sequence retrieval
		self.assertIs(self.tree(["branchA", "leafA"]),
		              self.tree("branchA")("leafA"),
		              msg="error in list retrieval")

		# string retrieval
		#print("asdafa")
		#print(self.tree("branchA/leafA"))
		#print(self.tree("branchA")("leafA"))
		self.assertEqual( self.tree(
			self.tree.separatorChars["child"].join(["branchA", "leafA"])),
			self.tree("branchA")("leafA"),
		                 msg="string address error")

		# malformed address retrieval
		self.assertIs(self.tree("branchA/leafA"),
			self.tree("branchA")("leafA"),
			msg="string address error")
		self.assertIs(self.tree("/branchA/leafA"),
		                 self.tree("branchA")("leafA"),
		                 msg="string address error")
		self.assertIs(self.tree("branchA/", "leafA"),
		                 self.tree("branchA")("leafA"),
		                 msg="string address error")
		self.assertIs(self.tree("branchA", "/leafA"),
		                 self.tree("branchA")("leafA"),
		                 msg="string address error")
		self.assertIs(self.tree("branchA//leafA"),
		                 self.tree("branchA")("leafA"),
		                 msg="string address error")
		self.assertIs(self.tree("branchA/", "/leafA"),
		                 self.tree("branchA")("leafA"),
		                 msg="string address error")

		# parent retrieval
		self.assertEqual( self.tree("branchA", "leafA", "superleafA", create=True),
		                  self.tree("branchA", "leafA", "superleafA",
		                            "^", "^", "leafA", "superleafA"))

	def test_treeLookupCreate(self):
		# self.assertRaises(LookupError, self.tree, "nonBranch")
		self.tree.lookupCreate = True
		self.assertEqual(self.tree.lookupCreate, True)
		self.assertIsInstance(self.tree("nonBranch"), TreeInterface)

		newBranch = self.tree("nonBranch")
		self.assertEqual(newBranch.lookupCreate, True)
		self.assertIsNone(newBranch.getAuxProperty(TreeInterface.AuxKeys.LookupCreate))

		self.tree.lookupCreate = False
		self.assertRaises(LookupError, self.tree, "nonBranch2")




	def test_treeAddresses(self):
		""" test address system
		check equivalence of list and string formats """
		# sequence address
		self.assertEqual(self.tree("branchA")("leafA").address(),
		                  ["branchA", "leafA"],
		                  msg="address sequence error")
		# string address
		self.assertEqual(self.tree("branchA")("leafA").stringAddress(),
		      self.tree.separatorChars["child"].join(["branchA", "leafA"]),
		                 msg="string address error")
		# # uid address
		# leafUidAddress = [self.tree("branchA").uid, self.tree("branchA", "leafA").uid]
		# self.assertEqual(leafUidAddress,
		#                  self.tree("branchA", "leafA").address(),
		#                  msg="uid address error"
		#                  )

	def test_treeRemoval(self):
		self.assertIs(self.tree("branchA").parent, self.tree)
		self.assertIn(self.tree("branchA"), self.tree.branches)
		removedBranch = self.tree("branchA").remove()
		self.assertIs(removedBranch.parent, None)
		self.assertNotIn(removedBranch, self.tree.branches)

	def test_treeInsertion(self):
		""" test inserting new branch"""
		newBranch = self.treeCls(name="newBranch", value=69)
		self.tree("branchA")("leafA").addChild(newBranch)
		self.assertIs(self.tree("branchA")("leafA")("newBranch"),
		              newBranch)

	def test_treeEquality(self):
		""" testing distinction between identity and equality """
		newBranch = self.tree("branchA", "new", create=True)
		self.assertTrue(newBranch in self.tree("branchA"),
		                msg="Tree does not contain its branch")
		newCopy = newBranch.copy()
		self.assertEqual(newBranch, newCopy,
		                 msg="branch and its copy are not equal")

		self.assertFalse(newBranch is newCopy,
		                 msg="branch IS its copy")

		# check even for copying uids
		newExactCopy = newBranch.copy(copyUid=True)
		self.assertEqual(newBranch, newExactCopy,
		                 msg="branch and its exact uid copy are not equal")
		self.assertFalse(newBranch is newExactCopy,
		                 msg="branch IS its exact uid copy")

		"""shallow copying unsupported for now as it invites more confusion
		over uniqueness and referencing"""
		#
		# newCopy = newBranch.__deepcopy__()
		# self.assertEqual(newBranch, newCopy,
		#                  msg="branch and its shallow copy are not equal")

		# # self.assertTrue(newBranch is self.tree("branchA"),
		# #                 msg="tree does not contain its branch")
		# self.assertFalse(newCopy is self.tree("branchA"),
		#                  msg="tree contains copy of its branch")

	def test_treeRenaming(self):
		"""test that branches can be renamed and update properly"""
		renameBranch = self.tree("branchA")("leafA")
		self.assertEqual("leafA", renameBranch.name)
		renameBranch.name = "leafNew"

		print("renameBranch", renameBranch)

		self.assertEqual("leafNew", renameBranch.name)
		self.assertTrue("leafNew" in renameBranch.parent.keys())
		self.assertIs(renameBranch, self.tree("branchA")("leafNew"))

	def test_treeValues(self):

		self.tree("newName", create=True).value = "newValue"
		self.assertEqual(self.tree("newName", create=True).value, "newValue")

		newVal = "secondValue"
		self.tree["newName"] = newVal
		self.assertEqual(self.tree["newName"], newVal)

		self.tree("newName", "a", "b", create=True)
		self.tree["newName", "a", "b"] = "deepVal"
		self.assertEqual(self.tree["newName", "a", "b"], "deepVal")


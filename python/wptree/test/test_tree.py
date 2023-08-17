
import os


import unittest

from wptree import Tree


jsonOutPath = os.path.sep.join(
	os.path.split(__file__ )[:-1]) + "testLog.json"

class CustomTreeType(Tree):
	pass

class TestMainTree(unittest.TestCase):
	""" test for main tree interface methods """

	def setUp(self):
		""" construct a basic test tree """

		self.tree = Tree(name="testRoot", value="tree root")
		# self.tree.debugOn = True
		self.tree("branchA", create=True).value = "first branch"
		self.tree("branchA", create=True)("leafA", create=True).value = "first leaf"
		self.tree("branchB", create=True).value = 2

		# self.serialisedTruth = {'?VALUE': 'tree root', '?CHILDREN': [{'?VALUE': 'first branch', '?CHILDREN': [{'?VALUE': 'first leaf', '?NAME': 'leafA'}], '?NAME': 'branchA'}, {'?VALUE': 2, '?NAME': 'branchB'}], '?NAME': 'testRoot',
		#                         '?FORMAT_VERSION': 0,}

	# def test_treeSetup(self):
	# 	self.tree = Tree(name="testRoot", value="tree root")
	# 	# self.tree.debugOn = True
	# 	self.tree("branchA", create=True).value = "first branch"
	# 	self.tree("branchA", create=True)("leafA", create=True).value = "first leaf"
	# 	self.tree("branchB", create=True).value = 2

		# self.serialisedTruth = {'?VALUE': 'tree root', '?CHILDREN': [{'?VALUE': 'first branch', '?CHILDREN': [{'?VALUE': 'first leaf', '?NAME': 'leafA'}], '?NAME': 'branchA'}, {'?VALUE': 2, '?NAME': 'branchB'}], '?NAME': 'testRoot',
		#                         '?FORMAT_VERSION': 0,}


	def test_treeInternals(self):

		baseTree = Tree("basicRoot")
		#self.assertEqual(baseTree._uidBranchMap(), {})

		baseBranch = baseTree("basicBranch", create=True)
		print()
		print(baseBranch, baseTree.branchMap, baseTree.branches)

		self.assertTrue("basicBranch" in baseTree.keys())
		self.assertTrue("basicBranch" in baseTree.branchMap)

		self.assertEqual({"basicBranch" : baseBranch}, baseTree.branchMap)
		self.assertEqual([baseBranch], baseTree.branches)

	#
	# def test_treeRoot(self):
	# 	""" test that tree objects find their root properly """
	# 	self.assertIs( self.tree.root, self.tree,
	# 	                  msg="tree root is not itself")
	# 	self.assertIs(self.tree("branchA").root, self.tree,
	# 	                  msg="tree branch finds incorrect root of "
	# 	                      "{}".format(self.tree))
	# 	self.assertIs(self.tree("branchA")("leafA").root, self.tree,
	# 	                  msg="tree leaf finds incorrect root of "
	# 	                      "{}".format(self.tree))
	#
	# def test_treeRetrieval(self):
	# 	""" test retrieving values and branches
	# 	 using different methods """
	# 	# token retrieval
	# 	self.assertIs(self.tree("branchA", "leafA"),
	# 	              self.tree("branchA")("leafA"),
	# 	              msg="error in token retrieval")
	# 	# sequence retrieval
	# 	# self.assertIs(self.tree(["branchA", "leafA"]),
	# 	#               self.tree("branchA")("leafA"),
	# 	#               msg="error in list retrieval")
	# 	# NOT USED YET
	#
	# 	# string retrieval
	# 	self.assertEqual( self.tree(
	# 		self.tree.separatorChar.join(["branchA", "leafA"])),
	# 		self.tree("branchA")("leafA"),
	# 	                 msg="string address error")
	#
	# 	# parent retrieval
	# 	self.assertEqual( self.tree("branchA", "leafA", "superleafA", create=True),
	# 	                  self.tree("branchA", "leafA", "superleafA",
	# 	                            "^", "^", "leafA", "superleafA"))
	#
	# def test_treeLookupCreate(self):
	# 	self.assertRaises(LookupError, self.tree, "nonBranch")
	# 	self.tree.lookupCreate = True
	# 	self.assertEqual(self.tree.lookupCreate, True)
	# 	self.assertIsInstance(self.tree("nonBranch"), Tree)
	#
	# 	newBranch = self.tree("nonBranch")
	# 	self.assertEqual(newBranch.lookupCreate, True)
	# 	self.assertIsNone(newBranch.getAuxProperty(Tree._LOOKUP_CREATE_KEY))
	#
	#
	# def test_treeAddresses(self):
	# 	""" test address system
	# 	check equivalence of list and string formats """
	# 	# sequence address
	# 	self.assertEqual(self.tree("branchA")("leafA").address(),
	# 	                  ["branchA", "leafA"],
	# 	                  msg="address sequence error")
	# 	# string address
	# 	self.assertEqual(self.tree("branchA")("leafA").stringAddress(),
	# 	      self.tree.separatorChar.join(["branchA", "leafA"]),
	# 	                 msg="string address error")
	# 	# uid address
	# 	leafUidAddress = [self.tree("branchA").uid, self.tree("branchA", "leafA").uid]
	# 	self.assertEqual(leafUidAddress,
	# 	                 self.tree("branchA", "leafA").address(uid=True),
	# 	                 msg="uid address error"
	# 	                 )
	#
	# def test_treeRemoval(self):
	# 	self.assertIs(self.tree("branchA").parent, self.tree)
	# 	self.assertIn(self.tree("branchA"), self.tree.branches)
	# 	removedBranch = self.tree("branchA").remove()
	# 	self.assertIs(removedBranch.parent, None)
	# 	self.assertNotIn(removedBranch, self.tree.branches)
	#
	# def test_treeInsertion(self):
	# 	""" test inserting new branch"""
	# 	newBranch = Tree(name="newBranch", value=69)
	# 	self.tree("branchA")("leafA").addChild(newBranch)
	# 	self.assertIs(self.tree("branchA")("leafA")("newBranch"),
	# 	              newBranch)
	#
	# def test_treeEquality(self):
	# 	""" testing distinction between identity and equality """
	# 	newBranch = self.tree("branchA", "new", create=True)
	# 	self.assertTrue(newBranch in self.tree("branchA"),
	# 	                msg="Tree does not contain its branch")
	# 	# newCopy = newBranch.__copy__()
	# 	# self.assertEqual(newBranch, newCopy,
	# 	#                  msg="branch and its shallow copy are not equal")
	#
	# 	"""shallow copying unsupported for now as it invites more confusion
	# 	over uniqueness and referencing"""
	#
	# 	newCopy = newBranch.__deepcopy__()
	# 	self.assertEqual(newBranch, newCopy,
	# 	                 msg="branch and its shallow copy are not equal")
	# 	self.assertFalse(newBranch is newCopy,
	# 	                 msg="branch IS its copy")
	# 	# self.assertTrue(newBranch is self.tree("branchA"),
	# 	#                 msg="tree does not contain its branch")
	# 	self.assertFalse(newCopy is self.tree("branchA"),
	# 	                 msg="tree contains copy of its branch")
	#
	# def test_treeRenaming(self):
	# 	"""test that branches can be renamed and update properly"""
	# 	renameBranch = self.tree("branchA")("leafA")
	# 	self.assertEqual("leafA", renameBranch.name)
	# 	renameBranch.name = "leafNew"
	#
	# 	print("renameBranch", renameBranch)
	#
	# 	self.assertEqual("leafNew", renameBranch.name)
	# 	self.assertTrue("leafNew" in renameBranch.parent.keys())
	# 	self.assertIs(renameBranch, self.tree("branchA")("leafNew"))
	#
	# def test_treeValues(self):
	#
	# 	self.tree("newName", create=True).value = "newValue"
	# 	self.assertEqual(self.tree("newName", create=True).value, "newValue")
	#
	# 	newVal = "secondValue"
	# 	self.tree["newName"] = newVal
	# 	self.assertEqual(self.tree["newName"], newVal)
	#
	# 	self.tree("newName", "a", "b", create=True)
	# 	self.tree["newName", "a", "b"] = "deepVal"
	# 	self.assertEqual(self.tree["newName", "a", "b"], "deepVal")
	#
	#
	#
	# def test_treeSerialisation(self):
	# 	""" test serialising tree to dict
	# 	should get more advanced testing here, serialisation
	# 	needs to support more stuff """
	#
	# 	# self.assertEqual(self.tree.serialise(), self.serialisedTruth,
	# 	#                  msg="mismatch in serialised data")
	# 	# restoreTree = self.tree.deserialise(self.tree.serialise())
	# 	# self.assertEqual(self.tree, restoreTree,
	# 	#                  msg="restored tree not equal to its source")
	#
	#
	# def test_treeRegeneration(self):
	# 	""" test regeneration from dict """
	# 	self.assertTrue(Tree.deserialise(self.tree.serialise()).isEquivalent(self.tree, includeBranches=False))
	#
	# def test_treeTyping(self):
	# 	""" test custom tree types in contiguous hierarchy """
	# 	self.tree.addChild(
	# 		CustomTreeType("customBranch", value=34535))
	# 	restoreTree = self.tree.deserialise(self.tree.serialise())
	# 	# self.assertEqual(self.tree, restoreTree,
	# 	#                  msg="restored custom tree not equal to its source")
	# 	self.assertIs(type(restoreTree("customBranch")), CustomTreeType,
	# 	              msg="restored custom type not equal to its source")
	#
	# def test_treeBreakPoints(self):
	# 	"""test that breakpoints work properly"""
	# 	breakBranch = self.tree("branchA")
	# 	breakBranch.setBreakpoint("testBreakPoint")
	# 	#self.assertEqual(breakBranch.breakTags, {"testBreakPoint"})
	#
	# 	self.assertIs(self.tree("branchA", "leafA").getRoot("testBreakPoint"), breakBranch)
	# 	self.tree("branchA").setDefaultBreakpoint("testBreakPoint")
	# 	self.assertIs(self.tree("branchA", "leafA").root, breakBranch)




	# def test_treeSubRegions(self):
	# 	"""test that a "main" breakpoint creates a sub-region in
	# 	the tree"""
	# 	breakBranch = self.tree("branchA")
	# 	breakBranch.setBreakTags("main")
	#
	# 	self.assertIs(self.tree("branchA", "leafA").root, breakBranch)
	# 	self.assertIs(breakBranch.root, breakBranch)
	#
	# 	self.assertIs(self.tree("branchA", "leafA").absoluteRoot, self.tree)

	# disabling subregion stuff for now, still not sure how breakpoints should work


if __name__ == '__main__':

	unittest.main()




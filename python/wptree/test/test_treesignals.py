

import unittest

from wptree.delta import *

class TestTreeSignals(unittest.TestCase):
	""" test tree delta detection and signals """

	def setUp(self):
		""" construct a basic test tree """
		self.tree = TreeInterface(name="testRoot", value="tree root")
		self.tree("branchA", create=True).value = "first branch"
		self.tree("branchA", create=True)("leafA", create=True).value = "first leaf"
		self.tree("branchB", create=True).value = 2


	def test_deltaTrackerCreation(self):
		self.tree.createDefaultStateTracker(attach=True)
		self.assertIsInstance(self.tree.stateTracker(), TreeStateTracker)

		# check retrieving delta tracker from branches
		self.assertIs(self.tree.branches[0].stateTracker(), self.tree.stateTracker())
		# but check that child private attribute is still none
		self.assertIsNone(self.tree.branches[0]._stateTracker)

	def test_deltaSignals(self):
		print("pre attach")
		print("")
		tracker = self.tree.createDefaultStateTracker(attach=True)
		print("")
		print("post attach")
		print("")
		def signal_fn(*args, **kwargs):
			print("signal fn", args, kwargs)

		tracker.deltasChanged.connect(signal_fn)
		self.tree.value = "new value"

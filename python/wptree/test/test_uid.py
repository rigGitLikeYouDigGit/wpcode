
"""checking any core uid operations
"""



import os

import unittest

from tree.lib.python import bitwiseXor
from tree.lib.object import element

jsonOutPath = os.path.sep.join(
	os.path.split(__file__ )[:-1]) + "testLog.json"


class TestUid(unittest.TestCase):
	""" test for main tree interface methods """

	def setUp(self):
		pass

	def test_uidElement(self):
		elA = uidelement.UidElement()
		elB = uidelement.UidElement()

		self.assertNotEqual(elA.uid, elB.uid)
		self.assertIs(elA.byUid(elB.uid), elB)
		self.assertIs(elA.byUid(elA.uid), elA)

	def test_bitwiseXOR(self):
		elA = uidelement.UidElement()
		elB = uidelement.UidElement()

		uidScrambled = bitwiseXor(elA.uid, elB.uid)

		print(elA.uid)
		print(elB.uid)
		# printing looks funny since this becomes invalid unicode characters
		print(uidScrambled)

		self.assertIsInstance(uidScrambled, str)
		self.assertEqual(len(uidScrambled), len(elA.uid))

		self.assertNotEqual(uidScrambled, elA.uid)
		self.assertNotEqual(uidScrambled, elB.uid)

		# check reversible

		unscrambled = bitwiseXor(uidScrambled, elB.uid)
		print(unscrambled)
		self.assertEqual(unscrambled, elA.uid)











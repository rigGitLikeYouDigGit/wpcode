
import pprint
import unittest

from wplib.serial.main import serialiseRecursive, deserialiseRecursive

from wptree import Tree

class TestNode(unittest.TestCase):
	""" test for basic Chimaera node evaluation and referencing """

	def test_serial(self):
		# obj = {
		# 	"nodeA" : {
		# 		"inputs" : {
		# 			"uid" : ["nodeB", "nodeC"],
		# 			"state" : "connected",
		# 		},
		# 	}
		# }
		# serialData = serialiseRecursive(obj)
		# deserialData = deserialiseRecursive(serialData)
		# self.assertEqual(obj, deserialData)
		t = Tree("root")
		#t("nodeA", create=True)
		serialData = t.serialise()
		pprint.pprint(serialData)
		deserialData = Tree.deserialise(serialData)
		self.assertIsInstance(deserialData, Tree)


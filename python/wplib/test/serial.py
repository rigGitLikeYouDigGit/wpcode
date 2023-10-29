
import pprint
import unittest


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
		t = Tree("root", )
		#t("nodeA", create=True)
		serialData = t.serialise()
		#pprint.pprint(serialData)
		#print(serialData)
		deserialData = Tree.deserialise(serialData)
		#print(deserialData)
		self.assertIsInstance(deserialData, Tree)
		self.assertEqual(t.name, deserialData.name)

		newSerialData = deserialData.serialise()
		self.assertEqual(str(serialData), str(newSerialData))

		# add branches
		t["a", "b"] = Tree("branchTree", value=[23, {"a" : 1}])
		serialData = t.serialise()
		#print(serialData)
		loadedTree = Tree.deserialise(serialData)
		#print(loadedTree)
		newSerialData = loadedTree.serialise()
		self.assertEqual(str(serialData), str(newSerialData))

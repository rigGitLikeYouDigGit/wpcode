
import pprint
import unittest

from collections import OrderedDict

from wplib.serial import SerialAdaptor, Serialisable

#from wptree import Tree

class CustomObj(Serialisable):
	def __init__(self, value):
		self.value = value

	# def encode(self, encodeParams:dict=None) ->dict:
	# 	return {"value" : self.value}

	# @classmethod
	# def decode(cls, serialData:dict, decodeParams:dict=None) ->"CustomObj":
	# 	return cls(serialData["value"])

	def encode(self, encodeParams:dict=None) ->dict:
		return self.value

	@classmethod
	def decode(cls, serialData: dict, decodeParams: dict = None) -> "CustomObj":
		return cls(serialData)

	def __eq__(self, other):
		return self.value == other.value



class TestNode(unittest.TestCase):
	""" test for basic Chimaera node evaluation and referencing """

	def _checkRoundTrip(self, obj):
		serialData = Serialisable.serialise(obj)
		deserialData = Serialisable.deserialise(serialData)
		self.assertEqual(obj, deserialData)
		return serialData

	def test_serialTuple(self):

		obj = ()
		self._checkRoundTrip(obj)
		obj = (1, 2, 3)
		self._checkRoundTrip(obj)

	def test_serialDict(self):
		obj = {}
		self._checkRoundTrip(obj)
		obj = {"a" : 1, "b" : 2}
		data = self._checkRoundTrip(obj)
		print(data)

	def test_serialList(self):
		obj = []
		self._checkRoundTrip(obj)
		obj = [1, 2, 3]
		self._checkRoundTrip(obj)

	def test_serialSet(self):
		obj = set()
		self._checkRoundTrip(obj)
		obj = {1, 2, 3}
		self._checkRoundTrip(obj)

	def test_builtinCompounds(self):
		obj = {
			"nodeA" : {
				"inputs" : {
					"uid" : ["nodeB", "nodeC"],
					"state" : "connected",
				},
			}
		}
		data = self._checkRoundTrip(obj)
		print(data)

	def test_customType(self):

		obj = CustomObj(22)
		self._checkRoundTrip(obj)


		# obj = {
		# 	"nodeA" : {
		# 		(22, "(nodeB, nodeC)"),
		# 		OrderedDict,
		# 		CustomObj,
		# 		CustomObj(CustomObj)
		# 	}
		# }
		#
		# obj = {}
		#
		# serialData = Serialisable.serialise(obj)
		#
		# print(serialData)




	def test_tree(self):



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

		pprint.pprint(newSerialData)


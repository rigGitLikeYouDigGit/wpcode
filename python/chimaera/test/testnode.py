
import pprint
import unittest

from wplib.object import DirtyGraph, DirtyNode

from wptree import Tree

from chimaera.core.node import ChimaeraNode
#from chimaera.core.graph import ChimaeraGraph
from chimaera.core.exegraph import ChimaeraDirtyGraph
from chimaera.core.plugnode import PlugNode


class TestNode(unittest.TestCase):
	""" test for basic Chimaera node evaluation and referencing """

	def test_node(self):
		graph = ChimaeraNode("graph")

		nodeA = graph.createNode("nodeA")
		nodeB = graph.createNode("nodeB")
		nodeC = graph.createNode("nodeC")

		nodeA.setRef("inputs", uid=(nodeB.uid, nodeC.uid))

		self.assertIn("uid", nodeA.getRef("inputs"))
		self.assertEqual(nodeA.getRef("inputs")["uid"], [nodeB.uid, nodeC.uid])

		# check that an empty node returns its resultParams as data
		self.assertIsInstance(nodeA.value(), Tree)
		self.assertEqual(nodeA.value().name, "root")

		# check that a node with a value returns that value
		nodeA.setValue(1)
		self.assertEqual(nodeA.value(), 1)

		# check node expression tokens
		nodeA.setValue("$name")

		print(nodeA.valueExp()._parsedStructure)

		# def printArgs(*args, **kwargs):
		# 	print("printArgs", args, kwargs)
		# 	return args, kwargs
		#
		# nodeA.setValue(printArgs)
		# result = nodeA.value()
		# print(result)
		#
		# nodeA.setValue("(): print('hello')")
		# result = nodeA.value()
		# print(result, type(result))


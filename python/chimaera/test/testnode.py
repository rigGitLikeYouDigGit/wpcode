
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

		self.assertEqual(nodeA.value(), "nodeA")

		nodeA.setValue("$name.upper()")
		self.assertEqual(nodeA.value(), "NODEA")

		nodeA.setValue( lambda node : 22)
		self.assertEqual(nodeA.value(), 22)


	class TestPlugNode(PlugNode):
		""" test for basic Chimaera node evaluation and referencing """


		@classmethod
		def defaultParams(cls, paramRoot:Tree) ->Tree:
			paramRoot("inputs", create=True)
			return paramRoot

		def compute(self, **kwargs) ->object:
			return str(self.node.refMap())

	def test_plugNode(self):
		graph = ChimaeraNode("graph")

		node = graph.createNode("nodeA", TestNode.TestPlugNode)
		self.assertIsInstance(node, ChimaeraNode)
		self.assertEqual(node.name(), "nodeA")

		construct = TestNode.TestPlugNode.create("nodeB", graph)
		self.assertIsInstance(construct, TestNode.TestPlugNode)
		self.assertIs(construct.node.fnSet(), construct)
		print(construct.node.fnSet())
		print(construct.node.value())
		self.assertIn("inputs", construct.node.resultParams().branchMap)


	def test_serialisation(self):
		graph = ChimaeraNode("graph")

		nodeA = graph.createNode("nodeA")
		serialData = nodeA.serialise()
		pprint.pp(serialData)




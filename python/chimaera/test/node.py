

from __future__ import annotations

import pprint
import typing as T

import unittest

from wplib import log
from chimaera import ChimaeraNode

class TestNode(unittest.TestCase):
	""" tests for basic chimaeraNode behaviour """

	def test_attrCopying(self):
		node = ChimaeraNode("test")
		node.nodes.incomingTreeRaw().serialise()
		node.nodes.resolve()

	def test_nodeTyping(self):
		"""Check that the type system and syntax is consistent -
		ChimaeraNode calls returning appropriate objects"""

		#self.assertRaises(TypeError, lambda: ChimaeraNode("test"))

		# initialisation from type
		node = ChimaeraNode("test")
		self.assertIsInstance(node, ChimaeraNode)
		self.assertIs(type(node),  ChimaeraNode)

		# globally unique instances
		node2 = ChimaeraNode(node)
		self.assertIs(node, node2)
		node3 = ChimaeraNode(node.tree)
		self.assertIs(node, node3)

		# no parent
		self.assertIs(node.parent(), node.defaultGraph())

		# no children
		self.assertEqual(len(node.children()), 0)


		# test adding child node
		child = ChimaeraNode("child")
		self.assertIs(child.parent(), node.defaultGraph())

		node.addNode(child)
		self.assertIs(child.parent(), node)
		self.assertEqual(len(node.children()), 1)

		# test derived node types
		class NewType(ChimaeraNode):

			@classmethod
			def typeName(cls) ->str:
				return "newType"
		newNode = NewType("newNode")
		self.assertIsInstance(newNode, NewType)
		self.assertIsInstance(newNode, ChimaeraNode)
		chimNode = ChimaeraNode(newNode)
		self.assertIs(chimNode, newNode)


	def test_nodeRetrieval(self):
		node = ChimaeraNode("test")
		node = ChimaeraNode("test")


	def test_nodeValue(self):
		"""check resolving basic values -
		we're not defining values directly?
		or maybe we are

		compute is not implicit, it's just the
		default defined value?

		node.value = "testValue" # sets defined value.
		simple, fluid
		internally sets node.value.defined().value = "testValue"

		value:
		- incoming tree composed
		-

		defined is ALWAYS the final override on whatever its attribute,
		if you define a value, you don't want that value
		to get immediately mutated by compute()

		for now compute is always run as special case,
		with defined value overlaid on result

		"""

		node = ChimaeraNode("test")
		node.value.defined().value = "testValue"
		#log(node.value.defined())
		#log(node.value, node.value.resolve())

		self.assertEqual(node.value.resolve().value, "testValue")
		self.assertEqual(node.value().value, "testValue")


	def test_nodeConnections(self):
		"""check that nodes can be connected together implicitly
		by adding UIDs to incoming trees
		"""

		a = ChimaeraNode("a")
		b = ChimaeraNode("b")
		out = ChimaeraNode("out")

		out.value.incomingTreeRaw().value = [
			a.uid, b.uid
		]

		self.assertEqual(len(out.value.incomingTreeRaw().value), 2)

		# uid connections should work globally
		#out.value.resolveIncomingTree().display()


	def test_nodeCompute(self):
		"""define a simple node type, compute stuff"""


	def test_nodeSerialisation(self):
		"""welcome to pain"""

		baseNode = ChimaeraNode("test")
		# node.value.defined().value = "testValue"
		# node.value.incomingTreeRaw().value = [
		# 	"test1", "test2"
		# ]
		serialData = baseNode.serialise()
		#pprint.pprint(serialData, sort_dicts=False)

		regen = ChimaeraNode.deserialise(serialData)
		self.assertIsInstance(regen, ChimaeraNode)





from __future__ import annotations
import typing as T

import unittest

from wplib import log
from chimaera import ChimaeraNode, NodeType

class TestNode(unittest.TestCase):
	""" tests for basic chimaeraNode behaviour """

	def test_nodeTyping(self):
		"""Check that the type system and syntax is consistent -
		NodeTypes returning ChimaeraNode objects, and so on."""

		#self.assertRaises(TypeError, lambda: ChimaeraNode("test"))

		# initialisation from type
		node = NodeType("test")
		self.assertIsInstance(node, ChimaeraNode)

		# globally unique instances
		node2 = ChimaeraNode(node)
		self.assertIs(node, node2)
		node3 = ChimaeraNode(node._data)
		self.assertIs(node, node3)

		# no parent
		self.assertIsNone(node.parent())

		# no children
		self.assertEqual(len(node.children()), 0)


		# test adding child node
		child = NodeType("child")
		self.assertIsNone(child.parent())

		node.addNode(child)
		self.assertIs(child.parent(), node)
		self.assertEqual(len(node.children()), 1)


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

		node = NodeType("test")
		node.value.defined().value = "testValue"
		log(node.value.defined())
		log(node.value, node.value.resolve())

		self.assertEqual(node.value.resolve().value, "testValue")
		self.assertEqual(node.value().value, "testValue")


	def test_nodeConnections(self):
		"""check that nodes can be connected together implicitly
		by adding UIDs to incoming trees
		"""

		a = NodeType("a")
		b = NodeType("b")
		out = NodeType("out")

		out.value.incomingTreeRaw().value = [
			a.uid, b.uid
		]

		self.assertEqual(len(out.value.incomingTreeRaw().value), 2)

		# uid connections should work globally
		out.value.incomingTreeResolved().display()


	def test_nodeCompute(self):
		"""define a simple node type, compute stuff"""






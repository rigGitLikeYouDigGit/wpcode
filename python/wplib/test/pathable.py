

from __future__ import annotations
import typing as T

import unittest

from collections import namedtuple

from wplib import log
from wplib.pathable import Pathable, DictPathable, IntPathable, SeqPathable


class TestPathable(unittest.TestCase):
	""" """

	def test_pathableTyping(self):
		"""check that type call returns the right
		subtype for given object"""

		self.assertIsInstance(Pathable({"a" : 1}), DictPathable)
		self.assertIsInstance(Pathable(1), IntPathable)
		self.assertIsInstance(Pathable([1, 2, 3]), SeqPathable)

	def test_dictPathable(self):

		obj = {"a" : 1, "b" : 2}
		pathable = DictPathable(obj)
		self.assertEqual(pathable["a"], 1)
		self.assertEqual(pathable["b"], 2)

		# more complex dict
		obj = {"a" : {"b" : 2}}
		pathable = DictPathable(obj)
		self.assertEqual(pathable["a", "b"], 2)
		# keys
		self.assertEqual(pathable["a", "keys()"], ["b"])

	def test_seqPathable(self):
		obj = [1, 2, 3]
		pathable = SeqPathable(obj)
		self.assertEqual(pathable[0], 1)
		self.assertEqual(pathable[1], 2)
		self.assertEqual(pathable[2], 3)

	def test_itemPaths(self):

		deepStructure = {
			"a" : {
				"b" : {
					"c" : 1
				},
				"d" : 2
			}
		}
		pathable = Pathable(deepStructure)
		self.assertEqual(pathable["a", "b", "c"], 1)
		# access pathable item
		item = pathable.access(pathable, ["a", "b", "c"], values=False, one=True)
		self.assertEqual(item.obj, 1)
		path = item.path()
		self.assertEqual(path, ["a", "b", "c"])

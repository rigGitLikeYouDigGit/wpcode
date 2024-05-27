

from __future__ import annotations
import typing as T

import unittest

from collections import namedtuple

from wplib import log
from wpdex import WpDex
from wpdex.dictdex import DictDex

class TestWPDex(unittest.TestCase):
	""" """

	def test_wpDexTyping(self):
		"""check that type call returns the right
		subtype for given object"""

		testDict = {"a":1, "b":2}
		obj = WpDex(testDict)
		self.assertIsInstance(obj, DictDex)

	def test_dictDex(self):

		obj = {"a" : 1, "b" : 2}
		pathable = WpDex(obj)
		self.assertEqual(pathable["a"], 1)
		self.assertEqual(pathable["b"], 2)

		# more complex dict
		obj = {"a" : {"b" : 2}}
		pathable = WpDex(obj)
		self.assertEqual(pathable["a", "b"], 2)
		# keys
		self.assertEqual(pathable["a", "keys()"], ["b"])

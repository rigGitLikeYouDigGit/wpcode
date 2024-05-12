

from __future__ import annotations
import typing as T

import unittest

from collections import namedtuple

from wplib import log
from wpdex import WpDex, DictDex

class TestWPDex(unittest.TestCase):
	""" """

	def test_wpDexTyping(self):
		"""check that type call returns the right
		subtype for given object"""

		testDict = {"a":1, "b":2}
		obj = WpDex(testDict)
		self.assertIsInstance(obj, DictDex)



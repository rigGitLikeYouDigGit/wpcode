
from __future__ import annotations
import typing as T

import unittest

from collections import namedtuple

from wplib import log
from wplib.object import DeepVisitor

TestNamedTuple = namedtuple("TestNamedTuple", ["a", "b", "c"])
VisitPassParams = DeepVisitor.VisitPassParams


class TestVisitor(unittest.TestCase):
	""" tests for basic chimaeraNode behaviour """

	def setUp(self):
		self.rawData = {
			"test": "test",
			"key": [1,2,3],
			"test2": TestNamedTuple(1,"two",3),
		}

	def test_visitor(self):
		"""basic function of visitor is just to iterate over a data structure"""

		rawData = self.rawData

		visitor = DeepVisitor()
		params = VisitPassParams()

		result = visitor.dispatchPass(
			rawData,
			params,
			visitFn=None
		)
		self.assertIsInstance(result, T.Generator)

		expectResult = (('test', 'test'), 'test', 'test', ('key', [1, 2, 3]), 'key', [1, 2, 3], 1, 2, 3, ('test2', TestNamedTuple(a=1, b="two", c=3)), 'test2', TestNamedTuple(a=1, b="two", c=3), 1, "two", 3)

		self.assertEqual(tuple(result), expectResult)


	def test_visitorApply(self):

		rawData = self.rawData

		visitor = DeepVisitor()
		params = VisitPassParams(
			visitFn=lambda *args: print("visitFn", args),

		)

		result = visitor.dispatchPass(
			rawData,
			params,
		)
		self.assertIsInstance(result, T.Generator)
		print("result", tuple(result))

	def test_visitorTransform(self):

		rawData = self.rawData
		visitor = DeepVisitor()

		def transformFn(obj, visitor, visitObjectData, visitPassParams):
			result = obj
			if isinstance(obj, str):
				result = obj.upper()
			#print("tf result", result)
			return result

		params = VisitPassParams(
			visitFn=transformFn,
			transformVisitedObjects=True,
		)

		result = visitor.dispatchPass(
			rawData,
			params,
		)
		print("result", result)


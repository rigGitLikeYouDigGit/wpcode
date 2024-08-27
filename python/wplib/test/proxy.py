
from __future__ import annotations
import typing as T

import unittest

from collections import namedtuple

from wplib import log
from wplib.object.proxy import Proxy, ProxyLink, LinkProxy, ProxyData


class TestProxy(unittest.TestCase):
	""" """


	def test_proxyTyping(self):
		"""check that type call returns the right
		subtype for given object"""
		baseObj = 2

		class TestIntType(int):
			pass
		testObj = TestIntType(2)

		self.assertEqual(testObj, baseObj)

		getProxy = Proxy.getProxy(baseObj)

		#log("test mro", TestIntType.__mro__, vars=0)
		#log("mro", type(getProxy).__mro__, vars=0)
		#log("test bases", TestIntType.__bases__, vars=0)
		#log("bases", type(getProxy).__bases__, vars=0)

		self.assertIsInstance(getProxy, type(baseObj))
		self.assertTrue(issubclass(type(getProxy), type(baseObj)))
		self.assertEqual(getProxy, getProxy)
		self.assertEqual(getProxy, baseObj)
		self.assertNotEqual(getProxy, 3)


		for i in (2, "teeeest", 6.0, (4, 5, 6)
		          ):
			self._testImmType(i)

	def _testImmType(self, baseObj):
		print("test imm type", baseObj)
		getProxy = Proxy.getProxy(baseObj)
		self.assertIsInstance(getProxy, type(baseObj))
		self.assertTrue(issubclass(type(getProxy), type(baseObj)))
		self.assertEqual(getProxy, getProxy)
		self.assertEqual(baseObj, getProxy)
		self.assertEqual(getProxy, baseObj)


from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import unittest

from wpm import Plug, om, cmds, getMPlug


class TestPlugs(unittest.TestCase):
	"""test typing / broadcasting logic for
	node plugs"""

	def setUp(self):
		cmds.file(n=1, force=1)

	def test_mPlugRetrieval(self):

		node = cmds.createNode("uvPin")
		strPlug = node + ".coordinate"

		mPlug = getMPlug(strPlug)
		self.assertFalse(mPlug.isNull)
		self.assertTrue(mPlug.isArray)

	def test_plugTreeRetrieval(self):
		node = cmds.createNode("uvPin")
		strPlug = node + ".coordinate"

		plug = Plug(strPlug)
		self.assertIsInstance(plug, Plug)

	def test_plugTreeSlicing(self):
		node = cmds.createNode("uvPin")
		strPlug = node + ".coordinate"

		plug = Plug(strPlug)

		sliceItems = plug[2:5]
		print(sliceItems)


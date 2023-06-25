"""Been having trouble keeping the dependencies correct in this package-
This module is for the simple bases of any higher types we implement
within the maya environment, mainly for use in instance checking
DO NOT IMPORT MAYA CMDS HERE
openmaya is probably fine though

consider this forward-declaration
with python characteristics
"""


from __future__ import annotations
from typing import List, Set, Dict, Callable, Tuple, Sequence, Union, TYPE_CHECKING
from functools import partial

class NodeBase(object):

	@property
	def node(self):
		raise NotImplementedError
	pass


class PlugBase(object):

	@property
	def MPlug(self):
		raise NotImplementedError
	pass


class CleanupResource:
	"""very very temp test to stop old uis, functions etc
	from hanging around when WPM is reloaded"""

	instances = []

	def __init__(self):
		self.instances.append(self)

	def _cleanup(self):
		"""override this in subclasses"""
		pass

	@classmethod
	def cleanupAll(cls):
		"""cleanup all instances"""
		for i in cls.instances:
			i._cleanup()
			del i
		cls.instances = []




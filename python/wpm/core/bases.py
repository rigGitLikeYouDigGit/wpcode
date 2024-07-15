"""Been having trouble keeping the dependencies correct in this package-
This module is for the simple bases of any higher types we implement
within the maya environment, mainly for use in instance checking
DO NOT IMPORT MAYA CMDS HERE
openmaya is probably fine though

consider this forward-declaration
with python characteristics

the alternative to bases would be a try/except to check for
the attributes in methods, but if/else seems slightly
faster
"""


from __future__ import annotations
import typing as T
from functools import partial

if T.TYPE_CHECKING:
	from .api import om

class NodeBase(object):

	@property
	def MObject(self):
		raise NotImplementedError

	@property
	def MFn(self)->om.MFnDependencyNode:
		raise NotImplementedError

	@property
	def name(self):
		return self.MFn.name()
	@property
	def pathStr(self)->str:
		return self.MFn.uniqueName()


class PlugBase(object):

	MPlug : om.MPlug

	# @property
	# def MPlug(self):
	# 	raise NotImplementedError
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




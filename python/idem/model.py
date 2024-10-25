

from __future__ import annotations

import types
import typing as T
from pathlib import Path

from wplib import Sentinel, log
from wplib.object import Signal
from wptree import Tree
from wpdex import *
from wp import Asset, constant

from .node import IdemGraph, MayaSessionNode

"""
a single idem instance targets a single chi file, and for now a single asset

an idem session may have multiple instances open in tabs, but those tabs are
overall

target .chi file - look up a default from asset
asset from that file 


Q: now that we're so advanced, is there really much stopping us from going
	full object-hierarchy instead of full container-hierarchy?
A: yes, because user-exposed data has to be editable by hand,
	that's more difficult if you have to express a custom object in strings
"""

class IdemSession(Modelled):

	@classmethod
	def dataT(cls):
		return Tree

	@classmethod
	def newDataModel(cls, **kwargs) ->dataT():
		root = Tree("idem")
		root["asset"] = None
		root["filePath"] = Path("plan.chi")
		root["graph"] = IdemGraph.create(name="graph")
		return root



###### TODO #######
""" I have no idea how the stuff below will work, but
# from sketches it might evidently let us build more
# complex behaviour, and give a more stable structure
for the path referencing
"""

class Field:
	""" TEST
	really just reinventing half of param

	BUT here the advantage is we don't declare fields in the context of parent
	types, we just define them as their own objects and place them in
	a tree wherever needed

	persistent slot in structure holding typed information

	PASS IN PARENT if needed to retrieve value

	field itself has no additional address?

	if we don't do it this way, and do EVERYTHING by paths, we still need to set
	all the rules, validation, signals etc but without strong access to the actual values
	so try active fields for now

	"""

	def __init__(self, valType:(type, tuple[type], types.FunctionType),
	             value=None, **kwargs ):
		self.valType = valType
		self._value = value

		self.valueChanged = Signal("valueChanged")

	def setValue(self, val):
		self._value = val
	def getValue(self):
		return self._value


class PathField(Field):

	def __init__(self, valType,
	             value=None,
	             **kwargs):
		super().__init__(valType, value, **kwargs)

DEFAULT_CHI = "strategy.chi"

def newModel():
	""" add """
	root = Tree("root")
	root["asset"] = AssetField(default=None,
	                           optional=False,
	                           single=True
	                           )
	#root["chiPath"] = PathField(default=lambda : root["asset"].value.diskPath() / "stategy.chi")
	root["chiPath"] = PathField(
		parentDir=ref("../asset").diskPath(),
		default="strategy.chi",
		conditions=[ref("v").in_(ref("../asset").diskPath().iterdir())]
	)

	# set default paths

	# # if we use functions and callbacks
	# root["asset"].valueChanged.connect(
	# 	lambda **kwargs : root["chiPath"].setValue(DEFAULT_CHI))
	#
	# # if we try something more declarative?
	# root["chiPath"].setDefault(DEFAULT_CHI)
	# # some way to 'lift' a regular python function into a pipeline to call later?
	# #root["asset"].setDefault(Asset.fromDiskPath(root["chiPath"]))
	#
	# root["asset"].setDefault(lambda **kwargs : Asset.fromDiskPath(root["chiPath"]))






	pass



from __future__ import annotations

import types
import typing as T
from pathlib import Path

from wplib import Sentinel, log
from wplib.object import Signal
from wplib.time import TimeBlock
from wplib.serial import serialise, deserialise
from wptree import Tree
from wpdex import *
from wp import Asset, constant

from idem.node import IdemGraph#, MayaSessionNode

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
	
^ now tackling the above with typed/augmented tree branches in node settings, where needed
"""

DEFAULT_CHI = "plan.chi"

def getDefaultAsset():
	return Asset.topAssets()[0]

class IdemSession(Modelled):
	"""single instance of Idem
	TODO:
		- DCC environment for this session object
		- support having no asset selected - currently snaps to whatever the default asset is
	"""

	@classmethod
	def dataT(cls):
		return Tree

	@classmethod
	def newDataModel(cls, **kwargs) ->dataT():
		root = Tree(kwargs.get("name", "idem"))
		root["asset"] = kwargs.get("asset", getDefaultAsset())
		root["filePath"] = Path("plan.chi")
		root["graph"] = IdemGraph.create(name="graph")
		return root

	#TODO: these methods will return static values I think - if needed,
	#   caller can wrap results directly in WpDexProxy and it should find its
	#   way back to where needed
	def asset(self)->Asset:
		return self.rawData()["asset"]

	def fullChiPath(self)->Path:
		return self.asset().diskPath() / self.data["filePath"]

	#region serialisation
	def saveSession(self, toPath:Path=None):
		"""serialises the current session to the given path, or to the currently
		selected asset/file if none given

		timing: with a single basic node, serialisation takes about 0.005 seconds

		flatbuffers : strongly typed, non-starter for general serialisation
		flexbuffers : dynamically typed, smaller in memory but slower (how much slower?) than flatbuffers
		protobuf : typing.Any
		capnproto: AnyPointer

		"""
		toPath = toPath or self.fullChiPath()
		log("begin saving scene")
		with TimeBlock() as t:
			data = self.serialise()
		log("saved scene in ", t.time)





	#endregion

if __name__ == '__main__':
	s = IdemSession.create(name="testIdem")
	# log("    ")
	# log(s, type(s), s.data, type(s.data))
	# value = s.data.ref("asset", "@V")
	# log(value.rx.value)

	caitAsset = Asset.fromPath("tempest/asset/character/cait")
	ursusAsset = Asset.fromPath("tempest/asset/character/ursus")
	# s.data["asset"] = caitAsset
	# log("retrieved", s.data["asset"]) # works

	#ref = s.data.ref("filePath", "@V") # this also messes up, INVESTIGATE
	ref = s.data.ref("asset", "@V") # TODO: set wpdex to refresh children properly
	log("ref", ref, ref.rx.value)
	ref.WRITE(ursusAsset)
	log("new ref", ref, ref.rx.value)



###### TODO #######
""" I have no idea how the stuff below will work, but
# from sketches it might evidently let us build more
# complex behaviour, and give a more stable structure
for the path referencing

sketches below were superceded by other parts of the current system:
	- reactive widgets to update ui when internal state changes
	- typed / augmented tree branches used when declaring user-facing structures
	- WpDex and WpDexController managing path-based overrides 
"""

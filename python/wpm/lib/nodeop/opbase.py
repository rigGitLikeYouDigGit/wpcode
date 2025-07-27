
from __future__ import annotations
import typing as T

from dataclasses import dataclass

from wplib import log

from wpm import om, cmds, createWN, WPlug, WN
from wpm.core import plug

termType = (WPlug, om.MPlug, tuple, list, float, int, str, bool)



class PlugTerm:
	"""wrapper for more fluid syntax around plug calculations
	Marco D'Ambros I don't care if Copilot autocompletes your name
	this will be better than the node calculator

	plugTerm to be passed a bundle of plugs and values,
	to be used as a step in a large calculation
	"""

	def __init__(self, value:termType):
		"""value can be a plug or a value"""
		if isinstance(value, PlugTerm): # copy
			value = value.value

		self.value = attr.tryConvertToMPlugs(value)

	def __add__(self, other:termType)->PlugTerm:
		return PlugTerm(Add(self, other).outPlugs)



class NodeOp:
	"""
	wrapper around atomic node operations - should allow fluid and
	readable syntax when setting up systems

	nodes and connections are built on init for now
	"""

	def getOuterDGModifier(self)->om.MDGModifier:
		"""return the outer DGModifier for this operation"""
		return self._dgMod

	def getNewDGModifier(self)->om.MDGModifier:
		"""return a new DGModifier for this operation"""
		return om.MDGModifier()

	def __init__(self):
		self.nodes:list[WN] = []
		self.outPlugs:list[WPlug] = [] # main "result" of node
		self.dgMod = self.getNewDGModifier()
		# existDgMod = self.getOuterDGModifier()
		# if existDgMod:
		# 	self.dgMod = existDgMod
		#

	def postBuild(self):
		"""run after building nodes -
		use to check if dgmod should run"""
		self.dgMod.doIt()


class Add(NodeOp):
	"""
	node operations contain steps of creating nodes,
	then connecting them
	"""
	def __init__(self, *terms):
		super().__init__()
		self.terms = terms
		self.buildNodes(terms[0], terms[1])

		self.postBuild()


	def buildNodes(self, plugA:termType, plugB:termType):
		"""add plugs to the node.
		no support yet for more than 2 terms"""
		dgMod = self.dgMod
		if isinstance(plugA, PlugTerm):
			plugA = plugA.value
		if isinstance(plugB, PlugTerm):
			plugB = plugB.value
		log("plugA", plugA)
		log("plugB", plugB)
		for i, (a, b) in enumerate(attr.plugTreePairs(plugA, plugB)):
			node = WN.create("addDoubleLinear", dgMod=dgMod)
			attr.use(a, node("input1"), _dgMod=dgMod)
			attr.use(b, node("input2"), _dgMod=dgMod)
			self.nodes.append(node)
			self.outPlugs.append(node("output"))


def termTest():

	cmds.file(new=True, f=True)

	tfA = WN.create('transform')
	tfB = WN.create('transform')
	tfC = WN.create('transform')

	tfA.translate.set(1, 2, 3)
	tfB.translate.set(-2, -2, -2)

	result = PlugTerm(tfA.translate) + PlugTerm(tfB.translate)
	attr.use(result.value, tfC.translate)




class Subtract(NodeOp):
	"""
	"""

	def buildNodes(self, plugA:WPlug, plugB:WPlug, dgMod:om.MDGModifier):
		"""add plugs to the node"""
		for i, (a, b) in enumerate(plugLeafPairs(plugA, plugB)):
			node = WN.create("plusMinusAverage", dgMod=dgMod)
			a.con(node("input1D[0]"), dgMod=dgMod)
			b.con(node("input1D[1]"), dgMod=dgMod)
			node.attr("operation").set(2)
			self.nodes.append(node)
			self.outPlugs.append(node("output1D"))







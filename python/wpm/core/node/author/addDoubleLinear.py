from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from ..gen.addDoubleLinear import AddDoubleLinear as GenAddDoubleLinear
import numpy as np
from wpm import cmds, om, WN, to, arr, Plug, PLUG_VAL


class AddDoubleLinear(GenAddDoubleLinear):
	"""set up initialiser to pass in source plugs"""

	def __init__(self, name:str,
	             # dgMod: om.MDGModifier,
	             # dagMod: om.MDagModifier, # have modifiers as a context outside of initialiser
	             a:PLUG_VAL=0, b:PLUG_VAL=0,
	             ):
		self.input1_ = a
		self.input2_ = b

	def output(self) ->(None, Plug):
		return self.output_


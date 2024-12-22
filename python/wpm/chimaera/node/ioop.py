from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from chimaera import ChimaeraNode
from wptree import Tree
from .base import MayaOp
"""
consider node status can have multiple warnings:
for input:
	- if no nodes found in scene
	- if no file found at path

for output:
	- if no nodes matching found in scene
	- if no data incoming 
"""

"""inheritance - should each DCC io node import from a common
chimaera base?
TODO
"""
class IoOp(MayaOp):
	"""node to import separate mesh file,
	or load live mesh data from scene

	also allow exposing data at this point in graph as
	chimaera asset output

	importing from file or exporting incoming data to asset branch
	also implies creating a representation of that in the maya group
	for this node - maya ops can only directly export from maya entities
	"""

	@classmethod
	def defaultSettings(cls, forNode:ChimaeraNode, inheritedTree:Tree) ->Tree:
		inheritedTree["file"] = ""
		inheritedTree["nodeExp"] = "" # set this on the active set node
		inheritedTree["output"] = False # if True, output to an entry with this node's name
		return inheritedTree






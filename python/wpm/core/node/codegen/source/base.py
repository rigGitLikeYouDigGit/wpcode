
from __future__ import annotations
import typing as T

import json, sys, os
from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from dataclasses import dataclass

import orjson

from wptree import Tree

from wpm import WN, om, cmds
from wpm.core import getMFn

from wptool.codegen import CodeGenProject

"""
base classes for nodes, plugs and others

test using mixin bases to denote inputs, formats etc
to let ide catch errors

"""

class PlugDescriptor:
	"""descriptor for plugs -
	declare whole attr hierarchy in one go"""
	def __init__(self, name:str, mfnType:type[om.MFnAttribute]):
		self.name = name
		self.mfnType = mfnType

	# TEMP get and set
	def __get__(self, instance, owner):
		return self.name
	# TEMP
	def __set__(self, instance, value):
		self.value = value


class NodeBase:
	"""base class for nodes"""


jsonPath = Path(__file__).parent / "nodeData.json"

def genNodes(jsonPath, targetDir):
	"""generate nodes from json data"""
	with open(jsonPath, "r") as f:
		nodeData = json.load(f)

	for nodeTypeName, nodeData in nodeData.items():
		print(nodeTypeName, nodeData)

		# create a class for each node type
		# with a class attribute for each plug



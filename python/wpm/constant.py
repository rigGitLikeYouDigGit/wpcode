
from __future__ import annotations
import typing as T

import json, sys, os
from pathlib import Path


from wplib.object import TypeNamespace
from wpm.core.cache import cmds, om, oma

WPM_ROOT_PATH = __file__.parent # top "wpm" dir, use this for relative paths

# config used in case other languages need to draw from values
with open(WPM_ROOT_PATH / "config.json", "r") as f:
	config = json.load(f)


# in preparation for getting a plugin namespace from AD
WP_PLUGIN_NAMESPACE = int(config["WP_PLUGIN_NAMESPACE"])
# we have 256 nodes in this namespace ready for allocation
# we assume we might create at most 200 c++ nodes (which will be already registered)
CPP_PLUGIN_MAX_ID = int(config["CPP_PLUGIN_MAX_ID"])
# first start id for our python nodes
PY_PLUGIN_START_ID = WP_PLUGIN_NAMESPACE + CPP_PLUGIN_MAX_ID + 1
# this does make ids more volatile to assign them in code - maybe this will be an issue



ONE_VECTOR = om.MVector(1, 1, 1)
ZERO_VECTOR = om.MVector(0, 0, 0)
X_AXIS = om.MVector(1, 0, 0)
Y_AXIS = om.MVector(0, 1, 0)
Z_AXIS = om.MVector(0, 0, 1)

class Space:
	"""namespace for space constants
	not making this a full type namespace for now"""
	OBJECT = om.MSpace.kObject
	TF = om.MSpace.kTransform
	WORLD = om.MSpace.kWorld


class Data(TypeNamespace):
	"""namespace for attribute data types
	maybe move this
	maybe rename this
	for now it's fine
	"""
	class _Base(TypeNamespace.base()):
		pass

	class Float(_Base):
		pass
	class Angle(_Base):
		pass
	class Time(_Base):
		pass
	class Double(_Base):
		pass
	class Int(_Base):
		pass
	class Short(_Base):
		pass
	class Long(_Base):
		pass
	class Bool(_Base):
		pass
	class String(_Base):
		pass
	class Char(_Base):
		pass
	class Matrix(_Base):
		pass

	# array types
	class FloatArray(_Base):
		pass
	class AngleArray(_Base):
		pass
	class DoubleArray(_Base):
		pass
	class IntArray(_Base):
		pass
	class VectorArray(_Base):
		pass
	class MatrixArray(_Base):
		pass
	class StringArray(_Base):
		pass

	# shape types
	class NurbsCurve(_Base):
		pass
	class NurbsSurface(_Base):
		pass
	class Mesh(_Base):
		pass
	class Lattice(_Base):
		pass
	class Subdiv(_Base):
		pass

	# other
	class Enum(_Base):
		pass
	class Message(_Base):
		pass







class ImportMode(TypeNamespace):
	"""gathering data either from scene or from a linked file"""
	class _Base(TypeNamespace.base()):
		pass
	class FromFile(_Base):
		pass
	class FromScene(_Base):
		pass

class GraphDirection(TypeNamespace):
	"""direction of graph traversal"""
	class _Base(TypeNamespace.base()):
		pass
	class Future(_Base):
		pass
	class History(_Base):
		pass

class GraphTraversal(TypeNamespace):
	class _Base(TypeNamespace.base()):
		pass
	class DepthFirst(_Base):
		pass
	class BreadthFirst(_Base):
		pass

class GraphLevel(TypeNamespace):
	class _Base(TypeNamespace.base()):
		pass
	class Node(_Base):
		pass
	class Plug(_Base):
		pass


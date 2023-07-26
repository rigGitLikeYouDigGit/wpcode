
from wplib.object import TypeNamespace
from wpm.core.cache import cmds, om, oma


ONE_VECTOR = om.MVector(1, 1, 1)
ZERO_VECTOR = om.MVector(0, 0, 0)
X_AXIS = om.MVector(1, 0, 0)
Y_AXIS = om.MVector(0, 1, 0)
Z_AXIS = om.MVector(0, 0, 1)

OBJ_SPACE = om.MSpace.kObject
WORLD_SPACE = om.MSpace.kWorld


# in preparation for getting a plugin namespace from AD
WP_PLUGIN_NAMESPACE = 0x80040
# we have 256 nodes in this namespace ready for allocation
# we assume we might create at most 200 c++ nodes (which will be already registered)
CPP_PLUGIN_MAX_ID = 200
# first start id for our python nodes
PY_PLUGIN_START_ID = WP_PLUGIN_NAMESPACE + CPP_PLUGIN_MAX_ID + 1
# this does make ids ore volatile to assign them in code - maybe this will be an issue




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


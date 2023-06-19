
from tree.lib.object import TypeNamespace
from .cache import cmds, om, oma


ONE_VECTOR = om.MVector(1, 1, 1)
ZERO_VECTOR = om.MVector(0, 0, 0)
X_AXIS = om.MVector(1, 0, 0)
Y_AXIS = om.MVector(0, 1, 0)
Z_AXIS = om.MVector(0, 0, 1)

OBJ_SPACE = om.MSpace.kObject
WORLD_SPACE = om.MSpace.kWorld

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



from __future__ import annotations
"""constants across all of wp"""

from pathlib import Path

from tree.lib.object import TypeNamespace


"""
all files on disk should have consistent orientation and scale

SCALE : 1 unit = 1 decimetre
ORIENTATION : X left, Y up, Z forward



"""


# path constants
WP_ROOT = Path(__file__).parent.parent.parent.parent
TEMPEST_ROOT = WP_ROOT / "tempest"
# work out how to get this from the environment when we need different projects lol
ASSET_ROOT = TEMPEST_ROOT / "asset"

TEST_TEMPEST_ROOT = WP_ROOT / "test_tempest"
# work out how to get this from the environment when we need different projects lol
TEST_ASSET_ROOT = TEST_TEMPEST_ROOT / "asset"

TESTING = False

ROOT_MARKER_NAME = "WEPRESENT_ROOT"

def pathIsInDatabase(path:Path)->bool:
	"""checks if a path is in the database - at each level of parent, check
	if sibling files include the root marker file"""

	while path != Path():
		if not path.exists():
			path = path.parent
			continue
		if path.joinpath(ROOT_MARKER_NAME).exists():
			return True
		path = path.parent
	return False

def setTesting(testing:bool):
	"""sets the testing flag"""
	global TESTING
	TESTING = testing
	if testing:
		print("TESTING = TRUE")
		print("Now targeting assets in test directory")
		print("")

	else:
		print("TESTING = FALSE")
		print("# NOW AFFECTING PRODUCTION ASSETS # ")
		print("")

def getAssetRoot()->Path:
	"""returns the asset root path"""
	if TESTING:
		return TEST_ASSET_ROOT
	return ASSET_ROOT

#print("wp init")
# print("ROOT_PATH", WP_ROOT)
# print("ASSET_ROOT_PATH", ASSET_ROOT)

class CurrentAssetProject(TypeNamespace):
	"""eventually load from a config"""

	class _Base(TypeNamespace.base()):
		"""base class for asset project"""
		root : Path = Path("NONE")

	class Test(_Base):
		"""test asset project"""
		root = TEST_ASSET_ROOT

	class Tempest(_Base):
		"""tempest asset project"""
		root = ASSET_ROOT

#region enums
# enum constants common across all tools
# sort alphabetically

class GeoElement(TypeNamespace):
	"""enum for geo elements"""

	class _Base(TypeNamespace.base()):
		"""base class for geo element"""
		pass

	class Point(_Base):
		"""point"""
		pass

	class Vertex(_Base):
		"""vertex"""
		pass

	class Edge(_Base):
		"""edge"""
		pass

	class Primitive(_Base):
		"""primitive"""
		pass

	class Detail(_Base):
		"""detail"""
		pass

	class HalfEdge(_Base):
		"""half edge"""
		pass

	class Cell(_Base):
		"""voxel or pixel"""
		pass


class IoMode(TypeNamespace):
	"""Input, Output, None or Both -
	wherever we have nodes, processes, tools etc
	"""
	class _Base(TypeNamespace.base()):

		colour = (0, 0, 0)
		pass
	class Input(_Base):
		colour = (0, 0.5, 1) # blue input
	class Output(_Base):
		colour = (1, 0.7, 0.5) # orange output
	class Neither(_Base):
		colour = (0.5, 0.5, 0.5)
	class Both(_Base):
		colour = (0.8, 0.2, 0.8) # purple

class Orient(TypeNamespace):
	"""horizontal or vertical orientation - may rename this
	if it ends up colliding with XYZ directions"""

	class _Base(TypeNamespace.base()):
		"""base class for orient"""
		pass
	class Horizontal(_Base):
		"""horizontal orientation"""
		pass
	class Vertical(_Base):
		"""vertical orientation"""
		pass


class Status(TypeNamespace):
	"""abstract status of a tool or variable -
	is everything fine, or are there issues?

	This could form some kind of StatusObject base class too,
	stay with enum for now, define some common pastel colours
	"""
	class _Base(TypeNamespace.base()):
		"""base class for status"""
		colour = (0, 0, 0)
		pass
	class Success(_Base):
		"""everything is fine"""
		colour = (0.5, 1, 0.5)
		pass
	class Warning(_Base):
		"""something is not quite right"""
		colour = (1, 1, 0.5)
		pass
	class Error(_Base):
		"""something is wrong"""
		colour = (1, 0.5, 0.5)
		pass


class UiResponsiveness(TypeNamespace):
	"""should ui send updates immediately as user changes values,
	only when "enter" or "accept" is pressed,
	or not at all?"""
	class _Base(TypeNamespace.base()):
		pass
	class Immediate(_Base):
		"""send updates immediately as user changes values"""
		pass
	class OnAccept(_Base):
		"""send updates when "enter" or "accept" is pressed"""
		pass
	class NoAction(_Base):
		"""do not send updates (wait for rest of tool to read this value)"""
		pass

#endregion enums



from __future__ import annotations
"""constants across all of wp"""

import typing as T

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

print("wp init")
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

# enum constants common across all tools
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


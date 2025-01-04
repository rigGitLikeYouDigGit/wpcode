from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object import AttrDict, Catalogue

from wp.w3d.data import CameraData
"""test for serialisable typed commands as dictionaries for Idem"""

if T.TYPE_CHECKING:
	from .session import DCCIdemSession

# something quite filthy to get pycharm hints to work
_tdOld = T.TypedDict
TypedDict = T.TypedDict
T.TypedDict = dict

class IdemCmd(T.TypedDict, Catalogue, AttrDict ):
	"""
	some cheeky dynamic dispatch when explicitly called with the base class -
	IdemCmd(dict) -> MySpecificCmdCls(dict
		looks up the right wrapper and returns it

	IdemBridgeCmd(dict) ->
		specifically coerces the dict to this cmd type
	"""

	t : str # type of the command
	s : tuple[int, str]
	r : bool # does this command need a response

	catalogue = {}

	# if not T.TYPE_CHECKING:
	def __new__(cls, *args, **kwargs):
		if cls is IdemCmd:
			d = dict(*args, **kwargs)
			if "t" in d:
				newCls = cls.getCatalogueCls(d["t"])
				assert newCls, f"Unknown Idem command type in params: {args, kwargs}"
				return newCls(d)
		d = dict.__new__(cls, *args, **kwargs)
		d["t"] = cls.__name__
		return d

	def __str__(self):
		return f"<{self.__class__.__name__}{str(dict(self))}>"

T.TypedDict = _tdOld

# idem connection commands
class ConnectToBridgeCmd(IdemCmd):
	pass
class ConnectToSessionCmd(IdemCmd):
	targetPort : int
	pass
class DisconnectBridgeCmd(IdemCmd):
	pass
class DisconnectSessionCmd(IdemCmd):
	targetPort : int
	pass
class HeartbeatCmd(IdemCmd):
	pass

class ReplicateDataCmd(IdemCmd):
	data : dict

# dcc commands
class DCCCmd(IdemCmd):

	# set at class level
	session : DCCIdemSession = None
	def execute(self):
		"""apply this command to the current dcc scene"""
		raise NotImplementedError

class SetCameraCmd(DCCCmd):
	data : CameraData

if __name__ == '__main__':

	print(IdemCmd({'t': 'ConnectToBridgeCmd', 's': [58378, 'bridge'], 'r': True}))


	test = IdemCmd(s=(3, "b"))
	print(test)
	print(isinstance(test, IdemCmd))
	print(isinstance(test, dict))

	class CustomCmd(IdemCmd):
		customKey : str

	test = CustomCmd({"my" : "key"}, customKey="customVal")
	test.customKey = "customValue"
	print(test)


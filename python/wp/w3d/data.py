from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from dataclasses import dataclass

import numpy as np

from wplib.object import TDefDict
"""
in domain, we would want to load this same kind of data, 
apply it / extract it with the same interface, 
but in a different way

in maya:
	lib.applyCameraData( camera node )?

"""

class WpData(TDefDict):
	"""what a useful base class
	common base for general data used to serialise or
	transfer between DCCs -
	should we use USD instead? maybe.
	but USD is less visible to transfer across
	"""

	def apply(self, *args, **kwargs):
		"""DCC-side method to apply this method in the scene
		(usually to a node or quantity passed as argument to this
		function)"""
		raise NotImplementedError

	@classmethod
	def gather(cls, *args, **kwargs):
		"""DCC-side method to extract instance of this data
		object from the given arguments"""
		raise NotImplementedError

class CameraData(WpData):
	"""class representing state of camera in 3d scene"""
	matrix: np.ndarray
	focalLength: float
	resolution: tuple[int, int]
	orbitFocus: np.ndarray


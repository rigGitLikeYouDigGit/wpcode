
from __future__ import annotations
import typing as T

import importlib

"""init file to be copied into the gen directory"""

# add any extra imports
try:
	if T.TYPE_CHECKING:
		from transform import Transform
except NameError:
	pass





if T.TYPE_CHECKING:
	class Catalogue:
		Transform:Transform = Transform

		pass

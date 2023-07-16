
from __future__ import annotations
import typing as T


class SerialAdapter:
	"""Helper class to be used with external types,
	defining our own rules for saving and loading.

	This outer type should also define the type's serial-UID,
	and hold all versioned encoders.
	"""



from __future__ import annotations
import typing as T

class DerivedMetaBase(type):
	"""marker metaclass for detecting derived metaclasses"""
	pass


def deriveMetaClsForCls(*clsBases:T.Tuple[T.Type]):
	"""if cls tries to inherit from multiple classes with different metaclass,
	error occurs - needs specific leaf metaclass to be defined.
	Ideally this would be done automatically, but some boilerplate is ok for now.

	Strength ordering follows the order of the bases given, which
	should match the order of the actual bases passed to the class
	definition.
	"""

	metaBases = [type(base) for base in clsBases] + [DerivedMetaBase]
	if type in metaBases:
		metaBases.remove(type)

	class _DerivedMeta(*metaBases):
		"""metaclass for deriving metaclass from multiple bases
		"""
		def __repr__(self):
			return f"DerivedMeta({metaBases})"

	return _DerivedMeta

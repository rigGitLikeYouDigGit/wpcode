from __future__ import annotations

from wplib.object.namespace import TypeNamespace

class Sentinel(TypeNamespace):
	"""sentinel values for various things,
	usually function default and empty arguments
	(since None may be a valid value)
	"""
	class _Base(TypeNamespace.base()):
		"""base class for sentinel"""
		pass
	class Default(_Base):
		"""no value or override given, use default"""
		pass
	class Empty(_Base):
		"""value is empty"""
		pass
	class FailToFind(_Base):
		"""lookup not found"""
		pass
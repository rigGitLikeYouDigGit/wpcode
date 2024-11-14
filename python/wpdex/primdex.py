
from __future__ import annotations
import typing as T
from wplib.log import log

from .base import WpDex
from .proxy import WX

# dexes for primitives
"""while we could easily have one dex type to take care of all
of these, it makes it easier in UI if we can specify which kind of primitive
is used here"""
class PrimDex(WpDex):
	#forTypes = (int, float, bool, type(None))

	def _buildBranchMap(self, **kwargs) ->dict[DexPathable.keyT, WpDex]:
		return {}

	def bookendChars(self) ->tuple[str, str]:
		return "", ""

class IntDex(PrimDex):
	forTypes = (int, )
class BoolDex(PrimDex):
	forTypes = (bool, )
class NoneDex(PrimDex):
	"""why_cant_you_just_be_normal

	this is actually playing with hot fire, there's no point trying to proxy
	None since "x is None" either doesn't work or loses all meaning anyway.

	It is useful to be able to WRITE() to None in a container, so
	using the same interface as normal types would be worthwhile.
	"""
	forTypes = (type(None), )

	def ref(self, *path:WpDex.pathT
	        )->WX:
		"""convenience - sanity to get a reference to this dex
		value directly, without acrobatics through WpDexProxy
		not on the calling end at least
		"""
		if path:
			raise WpDex.PathKeyError("can't path into None")
		#TODO: factor this properly with ref creation on the main proxy type

		ref = WX(None, _dexPath=(), _dex=self)

		# flag that it should dirty whenever this proxy's
		# value changes (on delta)
		def _setDirtyRxValue(*_, **__):
			"""need to make a temp closure because we can't
			easily set values as a function call"""
			ref.rx.value = "TEMP_VAL"
			ref.rx.value = None

		self.getEventSignal("main").connect(_setDirtyRxValue)

		def _onRefWrite(path, value):
			self.write(value)

		ref._kwargs["_writeSignal"].connect(_onRefWrite)
		return ref



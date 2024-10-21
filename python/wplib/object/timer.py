
from __future__ import annotations
import typing as T

import threading, math

from wplib.object.namespace import TypeNamespace

"""test for a way to do simple animations for numeric stuff

maybe we can extend this into a similar interface as the maya functions
to set keyframes
"""

class EasingCurve(TypeNamespace):
	class _Base(TypeNamespace._Base):
		def apply(self, x:float)->float:
			"""given a linear float range between 0-1, apply this
			curve to it"""
			return x

	class Linear(_Base): pass
	class EaseOutQuart(_Base):
		def apply(self, x:float) ->float:
			return 1 - math.pow(1 - x, 4)

def animate(
		states:list[float],

            ):
	pass





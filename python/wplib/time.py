from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from time import perf_counter
from time import sleep
from contextlib import contextmanager

"""timing context managers by Justin Dehorty on stackoverflow """
@contextmanager
def catchtime() -> T.Callable[[], float]:
	t1 = t2 = perf_counter()
	yield lambda: t2 - t1
	t2 = perf_counter()

# with catchtime() as t:
#     sleep(1)

class TimeBlock:
	"""TODO: a way to reuse the same context in different blocks?
		     eg to start/stop timer
	"""

	def __init__(self):
		self.start = None
	def __enter__(self):
		self.start = perf_counter()
		return self

	def __exit__(self, type, value, traceback):
		self.time = perf_counter() - self.start
		# self.readout = f'Time: {self.time:.3f} seconds'
		# print(self.readout)
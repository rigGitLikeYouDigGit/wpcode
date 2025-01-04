
from __future__ import annotations
import typing as T
from types import FrameType
import inspect, pprint, itertools as it


def fast_stack(max_depth: int = None):
	""" from user Kache on stackOverflow

	Fast alternative to `inspect.stack()`

	Use optional `max_depth` to limit search depth
	Based on: github.com/python/cpython/blob/3.11/Lib/inspect.py

	Compared to `inspect.stack()`:
	 * Does not read source files to load neighboring context
	 * Less accurate filename determination, still correct for most cases
	 * Does not compute 3.11+ code positions (PEP 657)

	Compare:

	In [3]: %timeit stack_depth(100, lambda: inspect.stack())
	67.7 ms ± 1.35 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

	In [4]: %timeit stack_depth(100, lambda: inspect.stack(0))
	22.7 ms ± 747 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

	In [5]: %timeit stack_depth(100, lambda: fast_stack())
	108 µs ± 180 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

	In [6]: %timeit stack_depth(100, lambda: fast_stack(10))
	14.1 µs ± 33.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
	"""
	def frame_infos(frame: FrameType | None):
		while frame := frame and frame.f_back:
			yield inspect.FrameInfo(
				frame,
				inspect.getfile(frame),
				frame.f_lineno,
				frame.f_code.co_name,
				None, None,
			)

	return list(it.islice(frame_infos(inspect.currentframe()), max_depth))

def getLineLink(file=None, line=None) ->str:
	"""return clickable link to file, to output to log"""
	return f'File "{file}", line {max(line, 1)}'.replace("\\", "/")

def log(*args, file=None, line=None, printMsg=True, vars=False, frames=False, framesUp=0, **kwargs):
	"""print and append link to line, for easier debugging"""
	frameId = 1 + framesUp
	if file is None:
		file = inspect.stack()[frameId].filename
	if line is None:
		line = inspect.stack()[frameId].lineno
	#string = f'File "{file}", line {max(line, 1)}'.replace("\\", "/")
	string = ""
	if vars:
		string += " \nVARS: " +pprint.pformat(inspect.stack()[frameId].frame.f_locals) + "\n^ "# + " \n^ " + str(inspect.stack()[1].frame.f_globals)
	if frames:
		n = 1
		for i in inspect.stack()[frameId:]:
			string += getLineLink(i.filename, i.lineno) + "\n" + "\t" * n
			n+=1
	else:
		string += getLineLink(file, line)

	if printMsg:
		print(*args, string, **kwargs)
	return string


def getDefinitionFileLine(obj:T.Any) ->T.Tuple[str, int]:
	"""get file and line number of definition of object"""
	return inspect.getsourcefile(obj), inspect.getsourcelines(obj)[1]


def getDefinitionStrLink(obj:T.Any) ->str:
	"""get link to line number of definition of object"""
	return getLineLink(*getDefinitionFileLine(obj))



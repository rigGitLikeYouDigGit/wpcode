
from __future__ import annotations
import typing as T

# standardising type checking forstrings vs lists vs maps
strTypes = (str, ) # check first
mapTypes = (dict, )
seqTypes = (list, tuple, set, )

# types that can be literally evaluated from strings
literalTypes = (
	str, bytes, bool, int, float, complex
)

# for packing function arguments in tuple
argsKwargsType = T.Tuple[T.Tuple, T.Dict[str, object]]


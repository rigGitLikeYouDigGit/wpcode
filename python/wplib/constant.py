
from __future__ import annotations
import typing as T

# standardising type checking for strings vs lists vs maps
STR_TYPES = (str,) # check first
MAP_TYPES = (dict,)
SEQ_TYPES = (list, tuple, set,)

# types that can be literally evaluated from strings
LITERAL_TYPES = (
	str, bytes, bool, int, float, complex
)

# types that can't be changed after creation
IMMUTABLE_TYPES = (
	str, bytes, bool, int, float, complex, tuple, frozenset
)

# for packing function arguments in tuple
argsKwargsType = T.Tuple[T.Tuple, T.Dict[str, object]]


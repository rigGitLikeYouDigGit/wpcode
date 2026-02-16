from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import keyword
import builtins
import orjson
from wplib.time import TimeBlock

print(builtins.__dict__.keys())
print(keyword.kwlist)


p = "C:/Users/ed/Documents/GitHub/wpcode/python/wpm/core/node/codegen/nodeData.json"


with TimeBlock("read"):
	with open(p, "rb") as f:
		data = orjson.loads(f.read())





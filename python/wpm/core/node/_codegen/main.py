from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm.core.node._codegen import gather, generate


def regenMayaNodeFiles():
	generate.resetGenDir()
	gather.gatherNodeData()
	generate.genNodes()

if __name__ == '__main__':
	pass


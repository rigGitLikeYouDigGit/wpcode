from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm.core.node.codegen import gather, generate


def regenMayaNodeFiles():
	generate.resetGenDir()
	gather.updateDataForNodes()
	generate.genNodes()


def generateNodeFiles(nodeTypeNames:T.List[str]):
	"""specific node updates - by default targeting plugin node data"""
	gather.updateDataForNodes(nodeTypeNames)
	generate.genNodes(nodeTypeNames,
	                  )

if __name__ == '__main__':
	pass


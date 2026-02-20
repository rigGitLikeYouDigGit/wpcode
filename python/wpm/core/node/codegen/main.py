from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

#from wpm.core.node.codegen import gather, generate


def regenMayaNodeFiles():
	from wpm.core.node.codegen import gather, generate
	generate.resetGenDir()
	gather.updateDataForNodes()
	generate.genNodes()


def generateNodeFiles(nodeTypeNames:T.List[str]):
	from wpm.core.node.codegen import gather, generate
	"""specific node updates - by default targeting plugin node data"""
	gather.updateDataForNodes(nodeTypeNames)
	generate.genNodes(nodeTypeNames,
	                  )

if __name__ == '__main__':
	#regenMayaNodeFiles()
	from wpm.core.node.codegen import generate
	generate.regenMayaNodeFiles()
	pass


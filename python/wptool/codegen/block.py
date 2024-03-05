
from __future__ import annotations
import typing as T

import pprint
from pathlib import Path
from dataclasses import dataclass

from wptree import Tree

"""we model a generated code file as a tree of blocks,
each block usually corresponding to a function
or class definition.

we assume that any raw code file is read to lines
before being processed
"""



blockBreaks = {
	"def",
	"class",
}

#@dataclass
class Block:
	"""a block of code in a code file"""
	def __init__(self,
	             lines:list[str],
	             ):
		self.lines = lines

	def __repr__(self):
		return ("Block(\n\t" +
		        "\n\t".join(self.lines) +
		        "\n)")




def splitLinesToBlocks(lines:list[str]):
	"""given a list of lines, return corresponding tree of blocks"""
	blocks = []
	currentBlock = []
	for line in lines:
		currentBlock.append(line)
		for blockBreak in blockBreaks:
			if line.startswith(blockBreak):
				if currentBlock:
					blocks.append(Block(currentBlock))
					currentBlock = []
				break
	if currentBlock:
		blocks.append(Block(currentBlock))
		currentBlock = []
	return blocks


if __name__ == '__main__':
	# run analysis on this file
	thisFile = Path(__file__)
	with open(thisFile, "r") as f:
		lines = f.readlines()
	pprint.pprint(splitLinesToBlocks(lines))


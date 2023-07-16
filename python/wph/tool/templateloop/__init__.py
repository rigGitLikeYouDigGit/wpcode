

from __future__ import annotations
import typing as T
from pathlib import Path
from dataclasses import dataclass

import hou
from wph import reloadWPH

"""normal houdini loops cannot cache data between iterations, so
things like procedural skinning are basically impossible.

This is a test to literally copy a template node network out for each iteration,
whenever loop parametres or template network change.

Construction is super slow, but a consistent loop pattern 
should evaluate much faster when done

"""


"""hda structure:

loop
	/template - editable
		/inputs set up for normal amount of aux data
		
	/parallel_subnet
		/copies of template running in parallel
		/merge
	/feedback_subnet
		/copies of template wired end to end
	/switches to control parallel/feedback


"""

# probably easier to generate entire node structure through python,
# rather than hybrid python/hda

@dataclass
class LoopParams:
	"""params for loop hda"""
	iterations:int = 1
	parallel:bool = True
	pieceAttr:str = "piece"
	pieceAttrType:str = "int"


def clearLoopHda(hda:hou.Node):
	"""clear out all children of loop hda, except template"""

	hda.node("parallel_subnet").destroy()
	hda.node("feedback_subnet").destroy()

def rebuildLoopHda(hda:hou.Node):
	"""main entrypoint to rebuild loop hda"""






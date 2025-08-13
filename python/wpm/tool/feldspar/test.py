
import typing as T

from tree.lib.path import Path
from edRig import ECA, EdNode, cmds, om

"""
2d for now in the yz plane

effect of motion now created as multiple separate hierarchies 
once first is known, with positions varied.
"""

scenePath = Path(r"F:\all_projects_desktop\common\edCode\edRig\maya\resource\linkageBasicScene.ma")
def testLinkagePlugin():
	"""new scene, reload plugin, load test scene, setup"""
	cmds.file(new=1, f=1)
	





def testLinkage(rootPoints:T.List[EdNode],
                midPOints:T.List[EdNode],
                endPoint:EdNode,
                targetCrv:EdNode):
	"""given list of locked, stationary root points,
	and a target trajectory curve,

	generate a linkage with an output point



	"""
	print(rootPoints, targetCrv)

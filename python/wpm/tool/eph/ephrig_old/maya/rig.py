"""big oof"""

from edRig.ephrig.rig import EphRig
from edRig import ECA, EdNode


class MEphRig(EphRig):

	def __init__(self):
		super(MEphRig, self).__init__()

		self.mainNode = None #type:EdNode

	def setMainNode(self, n):
		self.mainNode = EdNode(n)

	def saveToNode(self):
		"""serialise this rig and dump it to its main control node"""
		data = self.serialise()
		self.mainNode("dataString").set(str(data))

	def edgeVectors(self):
		return [(i[1].outTf.worldPos(), i[0].childTf.worldPos())
		        for i in self.groundGraph.edges]

	def edgeMap(self):
		edges = list(self.groundGraph.edges)
		edgeMap = {edges.index(i) : {"nodes" : i} for i in self.groundGraph.edges}
		for index, data in edgeMap.items():
			span = data["nodes"][1].childTf.worldPos() - \
			       data["nodes"][0].childTf.worldPos()
			data["span"] = span
		return edgeMap


	def edgeSpans(self):
		"""return of """





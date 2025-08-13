
"""mpx node representing feldspar network generated from mesh"""

from __future__ import annotations

from maya.api import OpenMaya as om

from edRig.maya.lib.plugin import *
from edRig.maya.tool.feldspar.lib.matrix import *
from edRig.maya.tool.feldspar.plate import Plate, PlateParamData
from edRig.maya.tool.feldspar.vertex import Vertex
from edRig.maya.tool.feldspar.plugin.lib import *


def maya_useNewAPI():
	pass

"""
setup node creates 


"""

class FeldsparSetupNode(om.MPxNode, PluginNodeTemplate):
	"""creates a feldspar vertex network directly from vertices"""
	# define everything
	kNodeId = om.MTypeId( 0xDAA3)
	kNodeName = "feldsparSetup"

	paramDataCls = PlateParamData


	@classmethod
	def initialiseNode(cls):
		"""add attributes"""
		msgFn = om.MFnMessageAttribute()
		cFn = om.MFnCompoundAttribute()
		tFn = om.MFnTypedAttribute()
		nFn = om.MFnNumericAttribute()

		# inputs
		cls.aVertex, cls.aVertexPos, cls.aVertexIndex = makeVertexAttr(array=True)
		cls.aBar, cls.aBarVertexA, cls.aBarVertexB, cls.aBarBindLength, cls.aBarTargetLength, cls.aBarLength = makeBarAttr(array=True)
		cls.aGroup, cls.aGroupVertexIndex, cls.aGroupMatrix, cls.aGroupFixed = makeRigidGroupAttr(array=True)

		cls.aTick = makeTickAttr(readable=True, writable=False, array=False)

		toAdd = [cls.aVertex, cls.aBar, cls.aGroup, cls.aTick]
		for i in toAdd:
			cls.addAttribute(i)
		drivers = [cls.aVertex, cls.aBar, cls.aGroup]

		cls.drivenMObjects = [cls.aTick,
		                      ]
		setAttributesAffect(drivers, cls.drivenMObjects, cls)

	def postConstructor(self):
		"""create the Plate object and bind to this node"""
		self.plate = Plate(fixed=False)
		self._vertexArr = np.zeros((1, 1)) # internal use between methods


	def vertexObjsFromPlug(self, rootPlug):
		"""gather list of vertex objects from plugs"""
		vertices = []

		for i in range(rootPlug.numElements()):
			arrElementPlug = rootPlug.elementByPhysicalIndex(i)
			pos = arrElementPlug.child(self.aVertexPos).asMDataHandle().asDouble3()
			#print("pos", pos)

			# set vertex index plug if possible
			indexPlug = arrElementPlug.child(self.aVertexIndex)
			#print("i", i)
			indexPlug.setInt(i)

			vertex = Vertex(self.plate, index=i, pos=np.array((*pos, 1.0)))
			vertices.append(vertex)
		#print("final vertices", vertices)
		return vertices

	def syncVertices(self):
		"""function to regenerate vertex list for this plate"""
		plug = om.MFnDependencyNode(self.thisMObject()).findPlug(self.aVertex, True)
		vertexObjs = self.vertexObjsFromPlug(rootPlug=plug)
		self.plate.vertices = vertexObjs

	# rebuild vertices when a connection is changed
	def connectionBroken(self, thisPlug, otherPlug, asSrc):
		self.syncVertices()

	def connectionMade(self, thisPlug, otherPlug, asSrc):
		self.syncVertices()

	# # only rebuild vertices when requested by main node bind() signal
	# # so much simpler
	# def bind(self, pPlug:om.MPlug, pData:om.MDataBlock):
	# 	self.syncVertices()

	def gatherParams(self, pPlug:om.MPlug, pData:om.MDataBlock) ->paramDataCls:
		return self.paramDataCls(
			fixed=pData.inputValue(self.aIsFixed).asBool(),
		)

	def applyParams(self, pPlug: om.MPlug, pData: om.MDataBlock,
	                paramData: paramDataCls):
		self.plate.params = paramData


	#region compute methods

	def gatherVertexPositions(self, pData):
		"""retrieve vertex positions and return them"""
		positions = np.ndarray((len(self.plate.vertices), 4))
		posDH = pData.inputArrayValue(self.aVertex)
		posDH.jumpToPhysicalElement(0)
		index = 0
		while not posDH.isDone():
			vertexDH = posDH.inputValue()

			pos = vertexDH.child(self.aVertexPos).asFloat3()
			positions[index, :] = (*pos, 1.0)

			index += 1
			posDH.next()

		return positions

	def evaluate(self, pPlug:om.MPlug, pData:om.MDataBlock,
	             globalVertexPositions):
		"""set plate rest matrix to average of all vertex positions
		then set local rest positions of vertices

		if fixed (eg not under sim control), set live plate matrix
		instead

		"""
		globalVectors = np.array(globalVertexPositions)
		# get position average for rest matrix
		restMat = averageMatrixFromVectors(globalVectors)
		if self.plate.params.fixed:
			self.plate.matrix = restMat
		self.plate.restMatrix = restMat
		# localise global vertex positions
		localVertexPositions = multMatrixVectorArray(
			np.linalg.inv(restMat), globalVectors)
		for i, vtx in enumerate(self.plate.vertices):
			vtx.pos[:] = localVertexPositions[i]

	def setOutputs(self, pPlug, pData):
		pData.outputValue(self.aUid).setString(self.plate.uid)
		vtxArrayDH = pData.outputArrayValue(self.aVertex)
		vtxArrayDH.jumpToPhysicalElement(0)
		i = 0
		for i in range(len(self.plate.vertices)):
			vtxArrayDH.outputValue().child(self.aVertexLocalPos).set3Float(
				*tuple(self.plate.vertices[i].pos[:3])
			)
			vtxArrayDH.outputValue().child(self.aVertexIndex).setInt(i)
			vtxArrayDH.next()

		om.MFnMatrixData(pData.outputValue(self.aMatrix).data()).set(
			om.MMatrix(self.plate.restMatrix))



	# plate compute
	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		""""""
		#print("plate compute")

		# early filtering
		if pData.isClean(pPlug):
			return True

		# gather and apply params
		paramData = self.gatherParams(pPlug, pData)
		self.applyParams(pPlug, pData, paramData)

		# update structure
		posArr = self.gatherVertexPositions(pData)
		self.evaluate(pPlug, pData, posArr)

		# set outputs
		self.setOutputs(pPlug, pData)

		# tick
		flipTickAttr(self.aTick, pData)

		# set clean
		for i in self.drivenMObjects:
			pData.setClean(i)
		pData.setClean(pPlug)
		return True




	#endregion














